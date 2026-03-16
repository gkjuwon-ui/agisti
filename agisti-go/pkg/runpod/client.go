/*
Package runpod provides a Go client for the RunPod GPU cloud API.

Replaces the Python urllib-based client with:
  - Connection pooling via http.Client
  - Concurrent pod management
  - Automatic retry with exponential backoff
  - Structured error types
*/
package runpod

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"time"
)

const (
	defaultBaseURL = "https://api.runpod.io/graphql"
	defaultTimeout = 30 * time.Second
	maxRetries     = 3
)

// Client is a thread-safe RunPod API client.
type Client struct {
	apiKey  string
	baseURL string
	http    *http.Client
}

// NewClient creates a RunPod client with connection pooling.
func NewClient(apiKey string) *Client {
	return &Client{
		apiKey:  apiKey,
		baseURL: defaultBaseURL,
		http: &http.Client{
			Timeout: defaultTimeout,
			Transport: &http.Transport{
				MaxIdleConns:        10,
				MaxIdleConnsPerHost: 5,
				IdleConnTimeout:     90 * time.Second,
			},
		},
	}
}

// Pod represents a RunPod GPU pod.
type Pod struct {
	ID          string  `json:"id"`
	Name        string  `json:"name"`
	Status      string  `json:"desiredStatus"`
	GPUType     string  `json:"machine>gpuDisplayName"`
	GPUCount    int     `json:"gpuCount"`
	CostPerHour float64 `json:"costPerHr"`
}

// CreatePodInput holds parameters for creating a pod.
type CreatePodInput struct {
	Name       string            `json:"name"`
	GPUTypeID  string            `json:"gpuTypeId"`
	GPUCount   int               `json:"gpuCount"`
	ImageName  string            `json:"imageName"`
	VolumeSizeGB int             `json:"volumeInGb"`
	Env        map[string]string `json:"env,omitempty"`
}

// CreatePod creates a new GPU pod on RunPod.
func (c *Client) CreatePod(ctx context.Context, input CreatePodInput) (*Pod, error) {
	query := `mutation($input: PodFindAndDeployOnDemandInput!) {
		podFindAndDeployOnDemand(input: $input) {
			id name desiredStatus gpuCount costPerHr
			machine { gpuDisplayName }
		}
	}`

	envSlice := make([]map[string]string, 0, len(input.Env))
	for k, v := range input.Env {
		envSlice = append(envSlice, map[string]string{"key": k, "value": v})
	}

	variables := map[string]interface{}{
		"input": map[string]interface{}{
			"name":              input.Name,
			"gpuTypeId":         input.GPUTypeID,
			"gpuCount":          input.GPUCount,
			"imageName":         input.ImageName,
			"volumeInGb":        input.VolumeSizeGB,
			"env":               envSlice,
			"cloudType":         "ALL",
			"supportPublicIp":   true,
			"startJupyter":      false,
			"startSsh":          true,
		},
	}

	resp, err := c.graphql(ctx, query, variables)
	if err != nil {
		return nil, fmt.Errorf("create pod: %w", err)
	}

	data, ok := resp["podFindAndDeployOnDemand"].(map[string]interface{})
	if !ok {
		return nil, fmt.Errorf("unexpected response format")
	}

	return parsePod(data), nil
}

// GetPod retrieves a pod's current status.
func (c *Client) GetPod(ctx context.Context, podID string) (*Pod, error) {
	query := `query($podId: String!) {
		pod(input: { podId: $podId }) {
			id name desiredStatus gpuCount costPerHr
			machine { gpuDisplayName }
		}
	}`

	resp, err := c.graphql(ctx, query, map[string]interface{}{
		"podId": podID,
	})
	if err != nil {
		return nil, fmt.Errorf("get pod %s: %w", podID, err)
	}

	data, ok := resp["pod"].(map[string]interface{})
	if !ok {
		return nil, fmt.Errorf("pod %s not found", podID)
	}

	return parsePod(data), nil
}

// StopPod stops a running pod.
func (c *Client) StopPod(ctx context.Context, podID string) error {
	query := `mutation($input: PodStopInput!) {
		podStop(input: $input) { id desiredStatus }
	}`

	_, err := c.graphql(ctx, query, map[string]interface{}{
		"input": map[string]string{"podId": podID},
	})
	if err != nil {
		return fmt.Errorf("stop pod %s: %w", podID, err)
	}
	return nil
}

// ListPods lists all pods.
func (c *Client) ListPods(ctx context.Context) ([]*Pod, error) {
	query := `query { myself { pods {
		id name desiredStatus gpuCount costPerHr
		machine { gpuDisplayName }
	}}}`

	resp, err := c.graphql(ctx, query, nil)
	if err != nil {
		return nil, fmt.Errorf("list pods: %w", err)
	}

	myself, ok := resp["myself"].(map[string]interface{})
	if !ok {
		return nil, fmt.Errorf("unexpected response format")
	}

	podsRaw, ok := myself["pods"].([]interface{})
	if !ok {
		return nil, fmt.Errorf("no pods in response")
	}

	pods := make([]*Pod, 0, len(podsRaw))
	for _, p := range podsRaw {
		m, ok := p.(map[string]interface{})
		if ok {
			pods = append(pods, parsePod(m))
		}
	}
	return pods, nil
}

func (c *Client) graphql(ctx context.Context, query string, variables map[string]interface{}) (map[string]interface{}, error) {
	body := map[string]interface{}{
		"query":     query,
		"variables": variables,
	}

	jsonBody, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}

	var lastErr error
	for attempt := 0; attempt < maxRetries; attempt++ {
		if attempt > 0 {
			backoff := time.Duration(1<<uint(attempt-1)) * time.Second
			log.Printf("[runpod] Retry %d/%d after %v", attempt+1, maxRetries, backoff)

			select {
			case <-time.After(backoff):
			case <-ctx.Done():
				return nil, ctx.Err()
			}
		}

		req, err := http.NewRequestWithContext(ctx, "POST", c.baseURL, bytes.NewReader(jsonBody))
		if err != nil {
			return nil, err
		}
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Authorization", "Bearer "+c.apiKey)

		resp, err := c.http.Do(req)
		if err != nil {
			lastErr = err
			continue
		}

		respBody, err := io.ReadAll(resp.Body)
		resp.Body.Close()
		if err != nil {
			lastErr = err
			continue
		}

		if resp.StatusCode >= 500 {
			lastErr = fmt.Errorf("server error: %d", resp.StatusCode)
			continue
		}

		if resp.StatusCode >= 400 {
			return nil, fmt.Errorf("client error %d: %s", resp.StatusCode, string(respBody))
		}

		var result struct {
			Data   map[string]interface{} `json:"data"`
			Errors []struct {
				Message string `json:"message"`
			} `json:"errors"`
		}

		if err := json.Unmarshal(respBody, &result); err != nil {
			return nil, fmt.Errorf("parse response: %w", err)
		}

		if len(result.Errors) > 0 {
			return nil, fmt.Errorf("graphql error: %s", result.Errors[0].Message)
		}

		return result.Data, nil
	}

	return nil, fmt.Errorf("exhausted retries: %w", lastErr)
}

func parsePod(data map[string]interface{}) *Pod {
	pod := &Pod{}
	if v, ok := data["id"].(string); ok {
		pod.ID = v
	}
	if v, ok := data["name"].(string); ok {
		pod.Name = v
	}
	if v, ok := data["desiredStatus"].(string); ok {
		pod.Status = v
	}
	if v, ok := data["gpuCount"].(float64); ok {
		pod.GPUCount = int(v)
	}
	if v, ok := data["costPerHr"].(float64); ok {
		pod.CostPerHour = v
	}
	if machine, ok := data["machine"].(map[string]interface{}); ok {
		if v, ok := machine["gpuDisplayName"].(string); ok {
			pod.GPUType = v
		}
	}
	return pod
}
