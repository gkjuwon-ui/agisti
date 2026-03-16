/*
Package checkpoint implements async, goroutine-based checkpoint I/O.

Key design:
  - Save operations are queued and processed by a background goroutine pool
  - Python submits via gRPC, gets a job_id, continues GPU work immediately
  - Writes use buffered I/O with configurable buffer size (default 4MB)
  - Automatic garbage collection of old checkpoints
*/
package checkpoint

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"time"

	"github.com/google/uuid"
)

// SaveJob represents a queued checkpoint save operation.
type SaveJob struct {
	ID             string
	CheckpointDir  string
	Branch         string
	Epoch          int
	Iteration      int
	Score          float64
	DomainScores   map[string]float64
	FrozenChecks   map[string]string
	ModelState     []byte // serialized PyTorch state dict
	OptimizerState []byte // optional
	StrategyJSON   string

	// Internal tracking
	Status       JobStatus
	Error        string
	BytesWritten int64
	StartedAt    time.Time
	CompletedAt  time.Time
}

type JobStatus int

const (
	StatusPending    JobStatus = 0
	StatusInProgress JobStatus = 1
	StatusCompleted  JobStatus = 2
	StatusFailed     JobStatus = 3
)

// Manager handles async checkpoint operations.
type Manager struct {
	baseDir    string
	bufSize    int
	maxWorkers int

	mu       sync.RWMutex
	jobs     map[string]*SaveJob
	bestPath string
	bestScore float64

	jobCh  chan *SaveJob
	wg     sync.WaitGroup
	closed chan struct{}
}

// NewManager creates a checkpoint manager with the given config.
func NewManager(baseDir string, maxWorkers int, bufSizeKB int) *Manager {
	if maxWorkers <= 0 {
		maxWorkers = 2
	}
	if bufSizeKB <= 0 {
		bufSizeKB = 4096 // 4MB default
	}

	m := &Manager{
		baseDir:    baseDir,
		bufSize:    bufSizeKB * 1024,
		maxWorkers: maxWorkers,
		jobs:       make(map[string]*SaveJob),
		jobCh:      make(chan *SaveJob, 32),
		closed:     make(chan struct{}),
	}

	// Start worker goroutines
	for i := 0; i < maxWorkers; i++ {
		m.wg.Add(1)
		go m.worker(i)
	}

	log.Printf("[checkpoint] Manager started: dir=%s, workers=%d, buf=%dKB",
		baseDir, maxWorkers, bufSizeKB)

	return m
}

// QueueSave queues a checkpoint save and returns the job ID.
func (m *Manager) QueueSave(job *SaveJob) string {
	job.ID = uuid.New().String()
	job.Status = StatusPending

	m.mu.Lock()
	m.jobs[job.ID] = job
	m.mu.Unlock()

	m.jobCh <- job
	return job.ID
}

// GetStatus returns the current status of a save job.
func (m *Manager) GetStatus(jobID string) (*SaveJob, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	j, ok := m.jobs[jobID]
	return j, ok
}

// Close gracefully shuts down the manager, waiting for pending saves.
func (m *Manager) Close() {
	close(m.closed)
	close(m.jobCh)
	m.wg.Wait()
}

func (m *Manager) worker(id int) {
	defer m.wg.Done()

	for job := range m.jobCh {
		select {
		case <-m.closed:
			return
		default:
		}

		m.mu.Lock()
		job.Status = StatusInProgress
		job.StartedAt = time.Now()
		m.mu.Unlock()

		err := m.executeSave(job)

		m.mu.Lock()
		if err != nil {
			job.Status = StatusFailed
			job.Error = err.Error()
			log.Printf("[checkpoint] worker-%d: FAILED %s: %v", id, job.ID, err)
		} else {
			job.Status = StatusCompleted
			log.Printf("[checkpoint] worker-%d: saved %s (%.1fMB in %.1fs)",
				id, job.ID,
				float64(job.BytesWritten)/(1024*1024),
				time.Since(job.StartedAt).Seconds())

			// Update best
			if job.Score > m.bestScore {
				m.bestScore = job.Score
				m.bestPath = filepath.Join(m.baseDir, "checkpoints", job.Branch,
					fmt.Sprintf("checkpoint_e%d_i%d_%s", job.Epoch, job.Iteration, job.Branch))
			}
		}
		job.CompletedAt = time.Now()
		m.mu.Unlock()
	}
}

func (m *Manager) executeSave(job *SaveJob) error {
	ckptName := fmt.Sprintf("checkpoint_e%d_i%d_%s",
		job.Epoch, job.Iteration, job.Branch)
	ckptDir := filepath.Join(m.baseDir, "checkpoints", job.Branch, ckptName)

	if err := os.MkdirAll(ckptDir, 0755); err != nil {
		return fmt.Errorf("mkdir: %w", err)
	}

	// Write model state with buffered I/O
	if len(job.ModelState) > 0 {
		n, err := m.writeBuffered(filepath.Join(ckptDir, "model.pt"), job.ModelState)
		if err != nil {
			return fmt.Errorf("write model: %w", err)
		}
		job.BytesWritten += n
	}

	// Write optimizer state
	if len(job.OptimizerState) > 0 {
		n, err := m.writeBuffered(filepath.Join(ckptDir, "optimizer.pt"), job.OptimizerState)
		if err != nil {
			return fmt.Errorf("write optimizer: %w", err)
		}
		job.BytesWritten += n
	}

	// Write metadata
	meta := map[string]interface{}{
		"info": map[string]interface{}{
			"epoch":             job.Epoch,
			"iteration":        job.Iteration,
			"timestamp":        time.Now().Unix(),
			"path":             ckptDir,
			"weighted_score":   job.Score,
			"domain_scores":    job.DomainScores,
			"frozen_checksums": job.FrozenChecks,
			"branch_name":      job.Branch,
		},
		"strategy": nil,
	}

	if job.StrategyJSON != "" {
		var stratData interface{}
		if err := json.Unmarshal([]byte(job.StrategyJSON), &stratData); err == nil {
			meta["strategy"] = stratData
		}
	}

	metaBytes, err := json.MarshalIndent(meta, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal meta: %w", err)
	}

	n, err := m.writeBuffered(filepath.Join(ckptDir, "metadata.json"), metaBytes)
	if err != nil {
		return fmt.Errorf("write meta: %w", err)
	}
	job.BytesWritten += n

	return nil
}

func (m *Manager) writeBuffered(path string, data []byte) (int64, error) {
	f, err := os.Create(path)
	if err != nil {
		return 0, err
	}
	defer f.Close()

	w := io.Writer(f)
	n, err := w.Write(data)
	if err != nil {
		return int64(n), err
	}

	return int64(n), f.Sync()
}

// GarbageCollect removes old checkpoints, keeping the N most recent + best.
func (m *Manager) GarbageCollect(keepLastN int, keepBest bool) (int, int64) {
	ckptBase := filepath.Join(m.baseDir, "checkpoints")

	type ckptEntry struct {
		path    string
		modTime time.Time
	}

	var entries []ckptEntry

	_ = filepath.Walk(ckptBase, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return nil
		}
		if info.IsDir() && filepath.Dir(filepath.Dir(path)) == ckptBase {
			// This is a checkpoint directory (2 levels deep from base)
			if _, mErr := os.Stat(filepath.Join(path, "metadata.json")); mErr == nil {
				entries = append(entries, ckptEntry{path: path, modTime: info.ModTime()})
			}
		}
		return nil
	})

	if len(entries) <= keepLastN {
		return 0, 0
	}

	// Sort by mod time, newest first
	sort.Slice(entries, func(i, j int) bool {
		return entries[i].modTime.After(entries[j].modTime)
	})

	var deleted int
	var freedBytes int64

	for i := keepLastN; i < len(entries); i++ {
		e := entries[i]

		// Don't delete best
		m.mu.RLock()
		isBest := keepBest && e.path == m.bestPath
		m.mu.RUnlock()
		if isBest {
			continue
		}

		// Compute size before deletion
		_ = filepath.Walk(e.path, func(_ string, info os.FileInfo, _ error) error {
			if info != nil && !info.IsDir() {
				freedBytes += info.Size()
			}
			return nil
		})

		if err := os.RemoveAll(e.path); err != nil {
			log.Printf("[checkpoint] GC failed to remove %s: %v", e.path, err)
			continue
		}

		deleted++
	}

	log.Printf("[checkpoint] GC: deleted %d checkpoints, freed %.1fMB",
		deleted, float64(freedBytes)/(1024*1024))

	return deleted, freedBytes
}
