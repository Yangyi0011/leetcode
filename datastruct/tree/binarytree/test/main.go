package main

import (
	"fmt"
	tree "leetcode/datastruct/tree/binarytree"
)
// ====================== 测试 =====================

/* 
	构建二叉树
	  0
    /   \
   1     2
  / \   /
 3   4  5

*/
func buildTree() *tree.TreeNode {
	node0 := &tree.TreeNode{Val : 0}
	node1 := &tree.TreeNode{Val : 1}
	node2 := &tree.TreeNode{Val : 2}
	node3 := &tree.TreeNode{Val : 3}
	node4 := &tree.TreeNode{Val : 4}
	node5 := &tree.TreeNode{Val : 5}

	node0.Left = node1
	node0.Right = node2
	node1.Left = node3
	node1.Right = node4
	node2.Left = node5
	return node0
}

// 先序遍历测试
// 先序: [0 1 3 4 2 5]
func preOrderTraversalTest() {
	root := buildTree()
	fmt.Println("先序:", tree.PreOrderTraversal(root))
}

// 中序遍历测试
// 中序: [3 1 4 0 5 2]
func inOrderTraversalTest() {
	root := buildTree()
	fmt.Println("中序:", tree.InOrderTraversal(root))
}

// 后序遍历测试
// 后序: [3 4 1 5 2 0]
func postOrderTraversalTest() {
	root := buildTree()
	fmt.Println("后序:", tree.PostOrderTraversal(root))
}

// 层级遍历测试
// 层级: [[3 4 5] [1 2] [0]]
func levelOrderTraversalTest() {
	root := buildTree()
	fmt.Println("层级:", tree.LevelOrderTraversal(root))
}

// Z型层级遍历测试
// Z型层级: [[0] [2 1] [3 4 5]]
func zigzagLevelOrderTest() {
	root := buildTree()
	fmt.Println("Z型层级:", tree.ZigzagLevelOrder(root))
}

func main() {
	preOrderTraversalTest()
	inOrderTraversalTest()
	postOrderTraversalTest()
	levelOrderTraversalTest()
	zigzagLevelOrderTest()
}
