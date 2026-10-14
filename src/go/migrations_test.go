package main

import (
	"testing"
	"testing/fstest"
)

func TestLoadMigrationsOrdersAndChecksumsFiles(t *testing.T) {
	files := fstest.MapFS{
		"migrations/000002_second.sql": {Data: []byte("SELECT 2;\n")},
		"migrations/000001_first.sql":  {Data: []byte("SELECT 1;\n")},
	}
	migrations, err := loadMigrations(files)
	if err != nil {
		t.Fatal(err)
	}
	if len(migrations) != 2 {
		t.Fatalf("migration count = %d, want 2", len(migrations))
	}
	if migrations[0].Version != 1 || migrations[0].Name != "first" {
		t.Fatalf("first migration = %#v", migrations[0])
	}
	if migrations[1].Version != 2 || migrations[1].Name != "second" {
		t.Fatalf("second migration = %#v", migrations[1])
	}
	if migrations[0].Checksum == "" || migrations[0].Checksum == migrations[1].Checksum {
		t.Fatalf("unexpected checksums: %#v", migrations)
	}
}

func TestLoadMigrationsRejectsDuplicateVersions(t *testing.T) {
	files := fstest.MapFS{
		"migrations/000001_first.sql": {Data: []byte("SELECT 1;")},
		"migrations/000001_again.sql": {Data: []byte("SELECT 2;")},
	}
	if _, err := loadMigrations(files); err == nil {
		t.Fatal("expected duplicate migration version to fail")
	}
}

func TestLoadMigrationsRejectsUnexpectedFiles(t *testing.T) {
	files := fstest.MapFS{
		"migrations/README.md": {Data: []byte("not a migration")},
	}
	if _, err := loadMigrations(files); err == nil {
		t.Fatal("expected invalid migration filename to fail")
	}
}
