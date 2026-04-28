package common

import (
	"os"
	"path/filepath"
	"sync"
	"time"
)

var (
	tempCounter uint64
	tempMu     sync.Mutex
)

func uniqueName(prefix, suffix string) string {
	tempMu.Lock()
	defer tempMu.Unlock()

	pid := os.Getpid()
	counter := tempCounter
	tempCounter++

	now := time.Now().UnixNano()
	return prefix + "_" + stringUint64(uint64(pid)) + "_" + stringUint64(uint64(now)) + "_" + stringUint64(counter) + suffix
}

type TempFile struct {
	FilePath    string
	DeleteOnDrop bool
}

func NewTempFile(suffix string) (*TempFile, error) {
	dir := os.TempDir()
	return NewTempFileIn(dir, suffix)
}

func NewTempFileIn(dir string, suffix string) (*TempFile, error) {
	name := uniqueName("bcc_tmp", suffix)
	path := filepath.Join(dir, name)

	file, err := os.Create(path)
	if err != nil {
		return nil, err
	}
	file.Close()

	return &TempFile{
		FilePath:    path,
		DeleteOnDrop: true,
	}, nil
}

func (tf *TempFile) Path() string {
	return tf.FilePath
}

func (tf *TempFile) Keep() string {
	tf.DeleteOnDrop = false
	return tf.FilePath
}

func (tf *TempFile) IntoPath() string {
	return tf.Keep()
}

func (tf *TempFile) Remove() error {
	if tf.DeleteOnDrop {
		return os.Remove(tf.FilePath)
	}
	return nil
}

type TempDir struct {
	DirPath     string
	DeleteOnDrop bool
}

func NewTempDir() (*TempDir, error) {
	dir := os.TempDir()
	return NewTempDirIn(dir)
}

func NewTempDirIn(parent string) (*TempDir, error) {
	name := uniqueName("bcc_dir", "")
	path := filepath.Join(parent, name)

	err := os.MkdirAll(path, 0755)
	if err != nil {
		return nil, err
	}

	return &TempDir{
		DirPath:     path,
		DeleteOnDrop: true,
	}, nil
}

func (td *TempDir) Path() string {
	return td.DirPath
}

func (td *TempDir) CreateFile(name string) (*TempFile, error) {
	path := filepath.Join(td.DirPath, name)

	file, err := os.Create(path)
	if err != nil {
		return nil, err
	}
	file.Close()

	return &TempFile{
		FilePath:    path,
		DeleteOnDrop: false,
	}, nil
}

func (td *TempDir) Keep() string {
	td.DeleteOnDrop = false
	return td.DirPath
}

func CreateTempObjectFile() (*TempFile, error) {
	return NewTempFile(".o")
}

func CreateTempAssemblyFile() (*TempFile, error) {
	return NewTempFile(".s")
}

func CreateTempPreprocessedFile() (*TempFile, error) {
	return NewTempFile(".i")
}

func stringUint64(i uint64) string {
	if i == 0 {
		return "0"
	}
	result := ""
	for i > 0 {
		result = string(rune('0'+i%10)) + result
		i /= 10
	}
	return result
}