package main

import (
	"fmt"
	"unsafe"
)

func main() {
	var a bool = true
	var b interface{} = nil
	// 1
	fmt.Println("size_a:", unsafe.Sizeof(a))
	// 16
	fmt.Println("size_b:", unsafe.Sizeof(b))
}