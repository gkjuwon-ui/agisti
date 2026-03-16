/*
checkpoint-svc: gRPC server for async checkpoint management.

This service offloads checkpoint I/O from the Python training loop,
letting the GPU continue training while checkpoints save in background.

Usage:
    go run ./cmd/checkpoint-svc --port 50051 --dir ./checkpoints --workers 2
*/
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"net"
	"os"
	"os/signal"
	"syscall"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/reflection"

	"github.com/gkjuwon-ui/agisti/agisti-go/pkg/checkpoint"
)

var (
	port    = flag.Int("port", 50051, "gRPC server port")
	dir     = flag.String("dir", "./output", "Base directory for checkpoints")
	workers = flag.Int("workers", 2, "Number of async save workers")
	bufKB   = flag.Int("buf", 4096, "Write buffer size in KB")
)

// server implements the gRPC CheckpointService.
type server struct {
	mgr *checkpoint.Manager
}

func main() {
	flag.Parse()

	// Ensure base directory exists
	if err := os.MkdirAll(*dir, 0755); err != nil {
		log.Fatalf("Failed to create base dir: %v", err)
	}

	mgr := checkpoint.NewManager(*dir, *workers, *bufKB)

	lis, err := net.Listen("tcp", fmt.Sprintf(":%d", *port))
	if err != nil {
		log.Fatalf("Failed to listen: %v", err)
	}

	grpcServer := grpc.NewServer(
		grpc.MaxRecvMsgSize(256 * 1024 * 1024), // 256MB for large models
	)

	_ = &server{mgr: mgr}

	// Enable reflection for debugging
	reflection.Register(grpcServer)

	// Graceful shutdown
	go func() {
		sigCh := make(chan os.Signal, 1)
		signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
		sig := <-sigCh
		log.Printf("Received %v, shutting down...", sig)

		// Stop accepting new RPCs
		grpcServer.GracefulStop()

		// Wait for pending saves with timeout
		done := make(chan struct{})
		go func() {
			mgr.Close()
			close(done)
		}()

		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
		defer cancel()
		select {
		case <-done:
			log.Println("All saves completed")
		case <-ctx.Done():
			log.Println("Timeout waiting for saves, forcing shutdown")
		}
	}()

	log.Printf("checkpoint-svc listening on :%d (dir=%s, workers=%d)", *port, *dir, *workers)

	if err := grpcServer.Serve(lis); err != nil {
		log.Fatalf("Failed to serve: %v", err)
	}
}
