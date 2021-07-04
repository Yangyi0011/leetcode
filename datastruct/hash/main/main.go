package main

import (
	"fmt"
	"strings"
)

type TreeNode struct {
    Val int
    Left *TreeNode
    Right *TreeNode
}
/* 
		 1
	   /   \
	  2     3
	 / \   / \
	4   5 6   7
*/
func buildTree() *TreeNode {
	node1 := &TreeNode{Val:1}
	node2 := &TreeNode{Val:2}
	node3 := &TreeNode{Val:3}
	node4 := &TreeNode{Val:4}
	node5 := &TreeNode{Val:5}
	node6 := &TreeNode{Val:6}
	node7 := &TreeNode{Val:7}

	node1.Left = node2
	node1.Right = node3
	node2.Left = node4
	node2.Right = node5
	node3.Left = node6
	node3.Right = node7
	return node1
}
func main() {
	root := buildTree()
	arr := preOrder(root)
	res := strings.Join(arr, ",")
	fmt.Println(res)
}

func preOrder(root *TreeNode) []string {
	result := make([]string, 0)
	if root == nil {
		return result
	}
	stack := make([]*TreeNode, 0)
	for root != nil || len(stack) > 0 {
		for root != nil {
			result = append(result, fmt.Sprintf("%d", root.Val))
			stack = append(stack, root)
			root = root.Left
		}
		result = append(result, "null")
		node := stack[len(stack) - 1]
		stack = stack[: len(stack) -1]
		root = node.Right
	}
	result = append(result, "null")
	return result
}

func inOrder(root *TreeNode) []string {
	result := make([]string, 0)
	if root == nil {
		return result
	}
	stack := make([]*TreeNode, 0)
	for root != nil || len(stack) > 0 {
		for root != nil {
			stack = append(stack, root)
			root = root.Left
		}
		result = append(result, "null")
		node := stack[len(stack) - 1]
		stack = stack[: len(stack) - 1]
		result = append(result, fmt.Sprintf("%d", node.Val))
		root = node.Right
	}
	result = append(result, "null")
	return result
}