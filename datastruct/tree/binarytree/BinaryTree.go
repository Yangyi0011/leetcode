// 前序遍历：先访问根节点，再前序遍历左子树，再前序遍历右子树 中序遍历：先中序遍历左子树，再访问根节点，再中序遍历右子树 后序遍历：先后序遍历左子树，再后序遍历右子树，再访问根节点
// 注意点
//     以根访问顺序决定是什么遍历
//     左子树都是优先右子树
package binarytree

// import (
// 	"fmt"
// )

type TreeNode struct {
	Val int
	Left *TreeNode
	Right *TreeNode
}

// ========================== 先序遍历 =============================
// 先序遍历对外函数
func PreOrderTraversal(root *TreeNode) []int {
	// 递归先序
	// result := make([]int, 0)
	// preOrderTraversalRecursion(root, &result)
	// return result
	
	// 循环先序
	// return preOrderTraversalLoop(root)

	// 深度优先先序
	// result := make([]int, 0)
	// preOrderDFS(root, &result)
	// return result

	// 分治法先序
	return preOrderDAC(root)
}

// 先序遍历（递归）
func preOrderTraversalRecursion(root *TreeNode, result *[]int) {
	// null & leaf
	if root == nil {
		return
	}
	*result = append(*result, root.Val)
	preOrderTraversalRecursion(root.Left, result)
	preOrderTraversalRecursion(root.Right, result)
}

// 先序遍历（循环）
func preOrderTraversalLoop(root *TreeNode) []int {
	result := make([]int, 0)
	if root == nil {
		return result
	}

	stack := make([]*TreeNode, 0)
	for root != nil || len(stack) > 0 {
		for root != nil {
			// Push
			stack = append(stack, root)
			result = append(result, root.Val)
			root = root.Left
		}
		// Pop
		node := stack[len(stack) - 1]
		stack = stack[:len(stack) - 1]
		root = node.Right
	}
	return result
}

// ========================== 中序遍历 =============================
// 中序遍历对外函数
func InOrderTraversal(root *TreeNode) []int {
	// 递归中序
	// result := make([]int, 0)
	// inOrderTraversalRecursion(root, &result)
	// return result

	// 循环中序
	return inOrderTraversalLoop(root)
}

// 中序遍历（递归）
func inOrderTraversalRecursion(root *TreeNode, result *[]int) {
	// null & leaf
	if root == nil {
		return
	}
	inOrderTraversalRecursion(root.Left, result)
	*result = append(*result, root.Val)
	inOrderTraversalRecursion(root.Right, result)
}

// 中序遍历（循环）
func inOrderTraversalLoop(root *TreeNode) []int {
	result := make([]int, 0)
	if root == nil {
		return result
	}

	stack := make([]*TreeNode, 0)
	for root != nil || len(stack) > 0 {
		for root != nil {
			stack = append(stack, root)
			root = root.Left
		}

		node := stack[len(stack) - 1]
		stack = stack[:len(stack) - 1]
		result = append(result, node.Val)
		root = node.Right
	}
	return result
}

// ========================== 后序遍历 =============================
func PostOrderTraversal(root *TreeNode) []int {
	// 递归后序
	// result := make([]int, 0)
	// postOrderTraversalRecursion(root, &result)
	// return result

	// 循环后序
	return postOrderTraversalLoop(root)
}

// 后序遍历（递归）
func postOrderTraversalRecursion(root *TreeNode, result *[]int) {
	// null & leaf
	if root == nil {
		return
	}
	postOrderTraversalRecursion(root.Left, result)
	postOrderTraversalRecursion(root.Right, result)
	*result = append(*result, root.Val)
}

// 后序遍历（循环）
func postOrderTraversalLoop(root *TreeNode) []int {
	result := make([]int, 0)
	if root == nil {
		return result
	}

	// 标记已弹出过的元素
	var isVisited *TreeNode
	stack := make([]*TreeNode, 0)
	for root != nil || len(stack) > 0 {
		for root != nil {
			stack = append(stack, root)
			root = root.Left
		}

		// 父节点必须在右节点弹出后再弹出
		node := stack[len(stack) - 1]
		// 当前节点的右节点为Null或当前节点的右节点已经弹出过才做处理
		// null：即当前节点是叶子节点，isVisited：即当前节点的右节点已经弹出过
		if node.Right == nil || node.Right == isVisited {
			result = append(result, node.Val)
			isVisited = node
			// Pop
			stack = stack[:len(stack) - 1]
		} else {
			root = node.Right
		}
	}
	return result
}

// ========================== 先序遍历DFS =============================
// 深度优先
func preOrderDFS(root *TreeNode, result *[]int) {
	if root == nil {
		return
	}
	*result = append(*result, root.Val)
	preOrderDFS(root.Left, result)
	preOrderDFS(root.Right, result)
}

// ========================== 先序遍历（分治法） =============================
// 从下到上
func preOrderDAC(root *TreeNode) []int {
	// null & leaf
	if root == nil {
		return []int{}
	}

	// divide
	left := preOrderDAC(root.Left)
	right := preOrderDAC(root.Right)

	// conquer
	result := make([]int, 0, len(left) + len(right) + 1)
	result = append(result, root.Val)
	result = append(result, left...)
	result = append(result, right...)
	return result
}

// ========================== 层级遍历 =============================
func LevelOrderTraversal(root *TreeNode) [][]int {
	// 从上到下
	// return levelOrderTraversal(root)

	// 从下到上
	return levelOrderTraversalReverse(root)
}

// 1、从上到下进行层级遍历
func levelOrderTraversal(root *TreeNode) [][]int {
	result := make([][]int, 0)
	if root == nil {
		return result
	}
	queue := []*TreeNode{root}
	for len(queue) > 0 {
		lenght := len(queue)
		level := make([]int, 0, lenght)
		// 处理当前层的元素，并由当前层的元素去确定下一层的元素
		for _, node := range queue {
			level = append(level, node.Val)
			if node.Left != nil {
				queue = append(queue, node.Left)
			}
			if node.Right != nil {
				queue = append(queue, node.Right)
			}
		}
		// 把当前层的元素值放入结果集
		result = append(result, level)
		// 去除已处理的当前层的元素
		queue = queue[lenght:]
	}
	return result
}

// 2、从下到上层级遍历
func levelOrderTraversalReverse(root *TreeNode) [][]int {
	result := levelOrderTraversal(root)
	// reverse
	for i := 0; i < len(result) / 2; i ++ {
		result[i], result[len(result) - 1 - i] = result[len(result) - 1 - i], result[i]
	}
	return result
}

// ========================== 锯齿形层次遍历 =============================
func ZigzagLevelOrder(root *TreeNode) [][]int {
	result := make([][]int, 0)
	if root == nil {
		return result
	}
	reverse := false
	queue := []*TreeNode{root}
	for len(queue) > 0 {
		lenght := len(queue)
		level := make([]int, 0, lenght)
		// 处理当前层的元素，并由当前层的元素去确定下一层的元素
		for _, node := range queue {
			level = append(level, node.Val)
			if node.Left != nil {
				queue = append(queue, node.Left)
			}
			if node.Right != nil {
				queue = append(queue, node.Right)
			}
		}
		if reverse {
			reverseList(level)
		}
		reverse = !reverse
		// 把当前层的元素值放入结果集
		result = append(result, level)
		// 去除已处理的当前层的元素
		queue = queue[lenght:]
	}
	return result
}
// 切片反转
func reverseList(list []int) {
	for i := 0; i < len(list) / 2; i ++ {
		list[i], list[len(list) - 1 - i] = list[len(list) - 1 - i], list[i]
	}
}