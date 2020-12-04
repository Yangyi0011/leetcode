package main

import (
	"fmt"
	"strconv"
)

func main() {
	// // 先序遍历
	// preorderTraversalTest()			// 递归
	// fmt.Println()
	// preorderLoopTest()				// 循环
	// preorderTraversalDFSTest()		// DFS-从上到下
	// preorderTraversalDACTest()		// 分治法-从下到上

	// // 中序遍历
	// inorderTraversalTest()			// 递归
	// fmt.Println()
	// inorderLoopTest()				// 循环

	// // 后续遍历
	// postorderTraversalTest()		// 递归
	// fmt.Println()
	postorderLoopTest()				// 循环

	// 层级遍历
	// levelOrderTest()				// BFS
}

/* 
	二叉树遍历
		先序遍历（根-左-右）：
			先访问根节点，再前序遍历左子树，再前序遍历右子树
		中序遍历（左-中-右）：
			先中序遍历左子树，再访问根节点，再中序遍历右子树
		后序遍历（左-右-中）：
			先后序遍历左子树，再后序遍历右子树，再访问根节点
	注意点
    	以根访问顺序决定是什么遍历
    	左子树都是优先右子树
*/
type TreeNode struct {
	Val int
	Left *TreeNode
	Right *TreeNode
}

func(this *TreeNode) String() string {
	str := strconv.Itoa(this.Val)
	return str
}

// 先序遍历（递归）
func preorderTraversal(root *TreeNode) {
	if root == nil {
		return
	}
	fmt.Printf("%v ", root.Val)
	preorderTraversal(root.Left)
	preorderTraversal(root.Right)
}

// 先序遍历（循环）
func preorderLoop(root *TreeNode) []int {
	if root == nil {
		return nil
	}

	result := make([]int, 0)
	stack := make([]*TreeNode, 0)

	for root != nil || len(stack) != 0 {
		for root != nil {
			// 先保存根节点的值
			result = append(result, root.Val)
			// 把根节点存入栈中，以便后续的右节点遍历
			stack = append(stack, root)

			// 遍历左节点
			root = root.Left
		}
		// 左分支遍历完后遍历右分支
		// 弹出保存的root节点遍历其右节点 Pop()
		node := stack[len(stack) -1]
		stack = stack[:len(stack) -1]
		root = node.Right
	}

	return result
}

// 中序遍历（递归）
func inorderTraversal(root *TreeNode){
	if root == nil {
		return
	}
	inorderTraversal(root.Left)
	fmt.Printf("%v ", root.Val)
	inorderTraversal(root.Right)
}

// 中序遍历（循环）
func inorderLoop(root *TreeNode) []int {
	if root == nil {
		return nil
	}

	result := make([]int, 0)
	stack := make([]*TreeNode, 0)

	for root != nil || len(stack) != 0 {
		for root != nil {
			// 把 root 节点放入栈中以便后续的操作
			stack = append(stack, root)
			// 深入左节点
			root = root.Left
		}
		// 出循环后 root 为nil
		
		// 从弹出一个 node，把它的值放入结果集中
		node := stack[len(stack) - 1]
		stack = stack[:len(stack) - 1]
		result = append(result, node.Val)

		root = node.Right
	}
	return result
}

// 后序遍历（递归）
func postorderTraversal(root *TreeNode) {
	if root == nil {
		return
	}

	postorderTraversal(root.Left)
	postorderTraversal(root.Right)
	fmt.Printf("%v ", root.Val)
}

// 后续遍历（循环）
// 核心就是：根节点必须在右节点弹出之后，再弹出
func postorderLoop(root *TreeNode) []int {
	if root == nil {
		return nil
	}

	result := make([]int, 0)
	stack := make([]*TreeNode, 0)

	// 通过lastVisit标识右子节点是否已经弹出
	var lastVisit * TreeNode
	for root != nil || len(stack) != 0 {
		for root != nil {
			// 将 root 节点放入栈中以便后续操作
			stack = append(stack, root)

			// 深入左节点
			root = root.Left
		}
		// 退出循环后 root == nil
		fmt.Println("stack1:", *stack)
		// 从栈中获取一个 node 节点，但不弹出
		node := stack[len(stack) - 1]
		// 根节点必须在右节点弹出之后再弹出
		if node.Right == nil || node.Right == lastVisit {
			stack = stack[:len(stack) - 1]  // 弹出Pop
			result = append(result, node.Val)

			// 标记当前这个节点已经弹出过
			lastVisit = node
		} else {
			// 深入右节点
			root = node.Right
		}
		fmt.Println("stack2:", *stack)
	}
	return result
}

// ==================== DFS(Depth First Search) =================
// DFS 深度搜索（从上到下） 和分治法（从下到上）区别：
// 前者一般将最终结果通过指针参数传入，后者一般递归返回结果最后合并

// 采用从上到下，DFS实现先序遍历（递归）
func preorderTraversalDFS(root *TreeNode) ([]int){
	// 结果集
	var result = make([]int, 0)
	DFS(root, &result)
	return result
}
// DFS-从上到下(递归)
func DFS(root *TreeNode, result *[]int) {
	if root == nil {
		return 
	}

	// 必须传入结果集的指针，否则这里append会产生新的对象，会造成结果数据丢失
	*result = append(*result, root.Val)
	DFS(root.Left, result)
	DFS(root.Right, result)
}

// 采用从下到上，分治法实现先序遍历（递归）
func preorderTraversalDAC(root *TreeNode) []int {
	return divideAndConquer(root)
}
// 分治法-从下到上（递归）
func divideAndConquer(root *TreeNode) []int {
	result := make([]int, 0)

	// 返回条件（null & leaf） leaf：叶子
	if root == nil {
		return result
	}

	// 分治（Divide）
	left := divideAndConquer(root.Left)
	right := divideAndConquer(root.Right)

	// 合并结果（Conquer）
	result = append(result, root.Val)
	result = append(result, left...)
	result = append(result, right...)
	return result
}

// ======================= BFS(Breadth First Seach) ============================
// 广度优先层级遍历
func levelOrder(root *TreeNode) [][]int {
	// 结果集
	result := make([][]int, 0)
	if root == nil {
		return result
	}

	// 创建一个队列，用于存放待处理层级的节点
	queue := make([]*TreeNode, 0)
	queue = append(queue, root)

	// 思路：遍历当前层的节点，通过当前层节点个数确定下一层的节点个数
	for len(queue) > 0 {
		// list存放当前层各个节点的值
		list := make([]int, 0)

		// 获取长度是为了遍历当前层的所有节点
		// 每处理完一个节点就把它的下一层节点添加进去（尾部追加）
		l := len(queue)
		for i := 0; i < l; i ++ {
			// 出队列
			level := queue[0]
			queue = queue[1:]
			list = append(list, level.Val)

			// 追加下一层节点
			if level.Left != nil {
				queue = append(queue, level.Left)
			}
			if level.Right != nil {
				queue = append(queue, level.Right)
			}
		}
		// 把当前层的数据放入结果集
		result = append(result, list)
	}
	return result
}





// ====================== 测试 ============================


/* 
	构建一棵树，结构如下：
		 0
	  1	    2
	3  4  5
*/
func buildTree() *TreeNode {
	var root = new(TreeNode)
	root.Val = 0

	node1, node2, node3, node4, node5 := new(TreeNode), new(TreeNode), new(TreeNode), new(TreeNode), new(TreeNode)
	node1.Val, node2.Val, node3.Val, node4.Val, node5.Val = 1, 2, 3, 4, 5

	root.Left = node1
	root.Right = node2
	node1.Left = node3
	node1.Right = node4
	node2.Right = node5

	return root
}

// 先序遍历（递归）测试
// 输出：0 1 3 4 2 5
func preorderTraversalTest() {
	root := buildTree()
	preorderTraversal(root)
}

// 先序遍历（循环）测试
// 输出： [0 1 3 4 2 5]
func preorderLoopTest() {
	root := buildTree()
	fmt.Println(preorderLoop(root))
}

// 中序遍历（递归）
// 输出：3 1 4 0 2 5
func inorderTraversalTest() {
	root := buildTree()
	inorderTraversal(root)
}

// 中序遍历（循环）
// 输出：[3 1 4 0 2 5]
func inorderLoopTest() {
	root := buildTree()
	fmt.Println(inorderLoop(root))
}

// 后序遍历（递归）
// 输出：3 4 1 5 2 0
func postorderTraversalTest() {
	root := buildTree()
	postorderTraversal(root)
}

// 后序遍历（循环）
// 输出：[3 4 1 5 2 0]
func postorderLoopTest() {
	root := buildTree()
	fmt.Println(postorderLoop(root))
}

// 先序遍历（递归DFS-从上到下）
// 输出：[0 1 3 4 2 5]
func preorderTraversalDFSTest() {
	root := buildTree()
	fmt.Println(preorderTraversalDFS(root))
}

// 先序遍历（递归分治法-从下到上）
// 输出：[0 1 3 4 2 5]
func preorderTraversalDACTest() {
	root := buildTree()
	fmt.Println(preorderTraversalDAC(root))
}

// 层级遍历（BFS）
// 输出：[[0] [1 2] [3 4 5]]
func levelOrderTest() {
	root := buildTree()
	fmt.Println(levelOrder(root))
}

