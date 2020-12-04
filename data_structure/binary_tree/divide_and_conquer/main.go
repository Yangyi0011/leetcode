package main

import (
	"fmt"
)

/* 
	分治法的应用：
		1、使用场景：
			·快速排序
			·归并排序
			·二叉树相关问题
			
		2、分治法模板
			（1）递归返回条件
			（2）分段处理
			（3）合并结果
			伪代码：
				func traversal(root * TreeNode) ResultType {
					// nil or leaf  
					// 树为空或是叶子节点
					if root == nil {
						// do something and return
						// 做一些条件判断，然后返回
					}

					// Divide
					// 分治
					ResultType left = traversal(root.Leaf)
					ResultType right = traversal(root.Right)

					// Conquer
					// 合并
					ResultType result = Merge from left and right

					return result
				}
*/

// ====================== 典型案列 ========================

// ============== （1）通过分治法遍历二叉树
type TreeNode struct {
	Val int
	Left *TreeNode
	Right *TreeNode
}

// 先序遍历
func preorderTraversalDFS(root *TreeNode) []int {
	return divideAndConquer(root)
}

// 分治法实现
func divideAndConquer(root *TreeNode) []int {
	result := make([]int, 0)
	
	// null or leaf
	if root == nil {
		return result
	}

	// divide
	left := divideAndConquer(root.Left)
	right := divideAndConquer(root.Right)

	// conquer
	result = append(result, root.Val)
	result = append(result, left...)
	result = append(result, right...)

	return result
}

// ======================== （2）归并排序
func MergeSort(nums []int) []int {
	return mergeSort(nums)
}

// 归并排序
// 注意点：递归需要返回结果用于合并
func mergeSort(nums []int) []int {
	l := len(nums)
	// 没有元素或只有一个元素时是有序的
	if l <= 1 {
		return nums
	}

	// divide：分为左右两段
	mid := l / 2
	left := mergeSort(nums[:mid])
	right := mergeSort(nums[mid:])

	// conquer：两段数据合并
	result := merge(left, right)
	return result
}

// 合并操作
func merge(left, right []int) []int {
	// 预先计算容量，避免扩容
	lLen, rLen := len(left), len(right)
	result := make([]int, 0, lLen + rLen)

	// 两边数组合并的游标
	l, r := 0, 0
	// 注意不能越界
	for l < lLen && r < rLen {
		// 谁小合并谁
		if left[l] < right[r] {
			result = append(result, left[l])
			l ++
		} else {
			result = append(result, right[r])
			r ++
		}
	}

	// 剩余的部分是有序的，直接合并
	result = append(result, left[l:]...)
	result = append(result, right[r:]...)

	return result
}

// ======================== （3）快速排序
/* 
	快速排序1：
	思路：
		把一个数组分为左右两段，左段小于右段，类似分治法没有合并过程

*/
func QuickSort(nums []int) []int {
	quickSort(nums, 0, len(nums)-1)
	return nums
}
// 原地交换，所以传入交换索引。切片是引用传递
func quickSort(nums []int, start, end int) {
	if start < end {
        // 分治法：divide，pivot：基准值的下标
		pivot := partition(nums, start, end)
		quickSort(nums, 0, pivot-1)
		quickSort(nums, pivot+1, end)
	}
}
// 分区
func partition(nums []int, start, end int) int {
	// 设置基准值
	p := nums[start]
	i, j := start, end

	// 过滤掉大于基准值的
	for i < j && nums[j] > p {
		j --
	}

	if i < j {
		nums[i], nums[j] = nums[j], nums[i]
		i ++
	}

	for i < j && nums[i] < p {
		i ++
	}

	if i < j {
		nums[i], nums[j] = nums[j], nums[i]
		j --
	}

	// p := nums[end]
	// i := start
	// for j := start; j < end; j++ {
	// 	if nums[j] < p {
	// 		swap(nums, i, j)
	// 		i++
	// 	}
	// }
    // // 把中间的值换为用于比较的基准值
	// swap(nums, i, end)

	// i ++, j -- 到最后必定会有 i == j, 此时的位置就是 基准值 所在的位置
	// 即完成了一轮快排
	return i
}
// 交换
func swap(nums []int, i, j int) {
	nums[i], nums[j] = nums[j], nums[i]
}

/* 
	快速排序2:
	思路：
		首先任意选取一个数据（通常选用数组的第一个数）作为关键数据（基准值），
		然后将所有比它小的数都放到它前面，所有比它大的数都放到它后面，
		这个过程称为一趟快速排序。
*/
func QuickSort2(nums []int) []int {
	quickSort2(nums, 0, len(nums)-1)
	return nums
}
// 快速排序方法二
func quickSort2(nums []int, start, end int) {
	
	i := start
	j := end

	// 基准值选左端，遍历就要先从右端开始，反之从左端开始
	key := nums[start]	// 选择第一个元素作为基准
	for i < j {
		// j -- 遍历，过滤掉大于等于 key 的数，循环会在遇到小于基值的数时停下来
		for i < j && nums[j] >= key {
			j --
		}

		// 若i >= j ，则不做任何处理，否则就是找到小于基值的数了
		if i < j {
			// nums[i] 的值已经保存在 key 里了，此处不用交换可以直接覆盖
			nums[i] = nums[j]

			// 交换后 i 往右移一位
			// 此时因为交换，基准值已经被换到 j 的位置了
			// 而 i 的位置已知是比基准值小的，所以对比的起点要改为 i + 1
			i ++
		}

		// i ++ 遍历，过滤掉小于等于 key 的数，循环会在遇到大于基值的数时停下来
		for i < j && nums[i] <= key {
			i ++
		}

		// 若i >= j ，则不做任何处理,否则就是找到大于基值的数了
		if i < j {
			// nums[j] 已经保存在上面的 nums[i] 中了，此处不用交换可以直接覆盖
			nums[j] = nums[i]

			// 交换过后 j 往左移一位
			// 此时因为交换，基准值已经被换到 i 的位置了
			// 而 j 的位置已知是比基准值大的，所以对比的起点要改为 j - 1
			j --
		}
	}
	
	// j --, i ++ 到之后一定会有 i == j，此时的位置就是基准所在的位置
	// 即完成了一轮快排
	nums[i] = key

	// 递归调用，把key前面的完成排序
	if i > start {
		quickSort2(nums, start, i - 1)
	}

	// 递归调用，把key后面的完成排序
	if j < end {
		quickSort2(nums, j + 1, end)
	}
}

/* 
======================= 二叉树的最大深度 ====================

给定一个二叉树，找出其最大深度。
二叉树的深度为根节点到最远叶子节点的最长路径上的节点数。

说明: 叶子节点是指没有子节点的节点。

示例：
给定二叉树 [3,9,20,null,null,15,7]，

    3
   / \
  9  20
    /  \
   15   7

返回它的最大深度 3 。

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/maximum-depth-of-binary-tree
*/
/*  
// 递归处理
func maxDepth(root *TreeNode) int {
    if root == nil {
        return 0
    }
    left := maxDepth(root.Left)
    right := maxDepth(root.Right)
    if left > right {
        return left + 1
    }
    return right + 1
} */

/* 
// DFS 处理
func maxDepth(root *TreeNode) int {
	var max *int = new(int)
	DFS(root, 0, max)
	return *max
}
func DFS(root *TreeNode, depth int, max *int) {
	if root == nil {
		if depth > *max {
			*max = depth
		}
		return
	}
	depth += 1
	DFS(root.Left, depth, max)
	DFS(root.Right, depth, max)
}
 */
// 通过BFS处理
func maxDepth(root *TreeNode) int {
    if root == nil {
        return 0
	}

	depth := 0
	queue := []*TreeNode{root}
	for len(queue) != 0 {
		depth ++
		length := len(queue)
		// 通过当前层级确认下一层级，当下一层级元素个数为0时，即到达了最深处
		for i := 0; i < length; i ++ {
			node := queue[i]
			if node.Left != nil {
				queue = append(queue, node.Left)
			}
			if node.Right != nil {
				queue = append(queue, node.Right)
			}
		}
		// 处理下一层级元素
		queue = queue[length:]
	}
    return depth
}


/* 
	============== 平衡二叉树 ================

给定一个二叉树，判断它是否是高度平衡的二叉树。
本题中，一棵高度平衡二叉树定义为：
    一个二叉树每个节点 的左右两个子树的高度差的绝对值不超过1。

示例 1:
给定二叉树 [3,9,20,null,null,15,7]

    3
   / \
  9  20
    /  \
   15   7

返回 true 。

示例 2:
给定二叉树 [1,2,2,3,3,null,null,4,4]

       1
      / \
     2   2
    / \
   3   3
  / \
 4   4

返回 false 。

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/balanced-binary-tree
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
*/

// 解法1：递归
func isBalanced(root *TreeNode) bool {
	_, b := depathAndBalance(root)
	return b
}
// 树是否平衡需要判断树的左右分支的高度差是否大于1
func depathAndBalance(root *TreeNode) (depath int, balance bool) {
	if root == nil {
		return 0, true
	}
	ld, lb := depathAndBalance(root.Left)
	rd, rb := depathAndBalance(root.Right)
	// 不平衡判断
	if lb == false || rb == false || abs(ld, rd) > 1 {
		balance = false
	} else {
		balance = true
	}
	depath = max(ld, rd) + 1
	return
}
func abs(a, b int) int {
	if a > b {
		return a - b
	}
	return b - a
}
func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

/* 
================= 二叉树中的最大路径和 ======================

给定一个非空二叉树，返回其最大路径和。
本题中，路径被定义为一条从树中任意节点出发，沿父节点-子节点连接，达到任意节点的序列。
该路径至少包含一个节点，且不一定经过根节点。

示例 1：
输入：[1,2,3]

       1
      / \
     2   3

输出：6

示例 2：
输入：[-10,9,20,null,null,15,7]

   -10
   / \
  9  20
    /  \
   15   7

输出：42

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/binary-tree-maximum-path-sum
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
*/
// 思路：分治法，分为三种情况：左子树最大路径和最大，右子树最大路径和最大，左右子树最大加根节点最大，
// 需要保存两个变量：一个保存子树最大路径和，一个保存左右加根节点和，然后比较这个两个变量选择最大值即可
func maxPathSum(root *TreeNode) int {
	return helper(root).MaxPath
}

type ResultType struct {
	// 保存单边最大值
	SinglePath int 	
	// 保存最大值（单边或两边+根的值）
	MaxPath int 	
}

func helper(root *TreeNode) ResultType {
	// check
	if root == nil {
		return ResultType {
			SinglePath: 0,
			MaxPath: -1<<31,
		}
	}

	// divide
	left := helper(root.Left)
	right := helper(root.Right)

	// conquer
	var result ResultType
	// 求单边最大值，0表示放弃该路径
	if left.SinglePath > right.SinglePath {
		result.SinglePath = max(left.SinglePath + root.Val, 0)
	} else {
		result.SinglePath = max(right.SinglePath + root.Val, 0)
	}

	// 求两边+根的最大值
	maxPath := max(left.MaxPath, right.MaxPath)
	result.MaxPath = max(maxPath, left.SinglePath + right.SinglePath + root.Val)
	return result
}


/* 
====================== 二叉树的最近公共祖先 ===================

给定一个二叉树, 找到该树中两个指定节点的最近公共祖先。
百度百科中最近公共祖先的定义为：“对于有根树 T 的两个结点 p、q，最近公共祖先表示为一个结点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（一个节点也可以是它自己的祖先）。”
例如，给定如下二叉树:  root = [3,5,1,6,2,0,8,null,null,7,4]
 
		3
	 /     \
	5       1
  /  \    /   \
 6   2   0     8
   /   \
  7     4

示例 1:
输入: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 1
输出: 3
解释: 节点 5 和节点 1 的最近公共祖先是节点 3。

示例 2:
输入: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 4
输出: 5
解释: 节点 5 和节点 4 的最近公共祖先是节点 5。因为根据定义最近公共祖先节点可以为节点本身。

说明:

    所有节点的值都是唯一的。
    p、q 为不同节点且均存在于给定的二叉树中。

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/lowest-common-ancestor-of-a-binary-tree
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
*/
// 思路：分治法，指定节点是否在当前节点的左子树或右子树中，在就返回当前节点，否则返回null
// func lowestCommonAncestor(root, p, q *TreeNode) *TreeNode {
// 	// check
// 	if root == nil {
// 		return nil
// 	}

// 	// 找到指定节点，直接返回
// 	if root == p || root == q {
// 		return root
// 	}

// 	// divide
// 	left := lowestCommonAncestor(root.Left, p, q)
// 	right := lowestCommonAncestor(root.Right, p, q)

// 	// conquer
// 	// 左右两边都不为空，则当前节点为公共祖先
// 	if left != nil && right != nil {
// 		return root
// 	}
// 	if left == nil {
// 		return right
// 	}
// 	if right == nil {
// 		return left
// 	}
// 	return nil
// }

// 思路：哈希表存储父节点，
// 1、从根节点开始遍历整棵二叉树，用哈希表记录每个节点的父节点指针。
// 2、从 p 节点开始不断往它的祖先移动，并用数据结构记录已经访问过的祖先节点。
// 3、同样，我们再从 q 节点开始不断往它的祖先移动，如果有祖先已经被访问过，即意味着这是 p 和 q 的深度最深的公共祖先，即 LCA 节点。
func lowestCommonAncestor(root, p, q *TreeNode) *TreeNode {
	if root == nil {
		return nil
	}

	parent := make(map[int] *TreeNode)
	visited := make(map[int] bool)

	var dfs func(*TreeNode)
	dfs = func(r *TreeNode) {
		if r == nil {
			return
		}
		// 用子节点的值当做父节点的key
		if r.Left != nil {
			parent[r.Left.Val] = r
			dfs(r.Left)
		}
		if r.Right != nil {
			parent[r.Right.Val] = r
			dfs(r.Right)
		}
	}

	dfs(root)

	for p != nil {
		visited[p.Val] = true
		// 访问p的父类
		p = parent[p.Val]
	}

	for q != nil {
		// 若被p标记为已访问，则说明是p、q的公共祖先
		if visited[q.Val] {
			return q
		}
		q = parent[q.Val]
	}
	return nil
}


/* 
=============================== 二叉树的层次遍历 =======================

给你一个二叉树，请你返回其按 层序遍历 得到的节点值。 （即逐层地，从左到右访问所有节点）。

示例：
二叉树：[3,9,20,null,null,15,7],

    3
   / \
  9  20
    /  \
   15   7

返回其层次遍历结果：
[
  [3],
  [9,20],
  [15,7]
]

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/binary-tree-level-order-traversal
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
*/
func levelOrder(root *TreeNode) [][]int {
	result := make([][]int, 0)
	if root == nil {
		return result
	}
	queue := []*TreeNode{root}
	for len(queue) > 0 {
		length := len(queue)
		level := make([]int, 0, length)
		// 处理当前层级的元素，并通过当前层级把下一层级的元素入队
		for _, node := range queue {
			level = append(level, node.Val)
			if node.Left != nil {
				queue = append(queue, node.Left)
			}
			if node.Right != nil {
				queue = append(queue, node.Right)
			}
		}
		result = append(result, level)
		// 已处理过的当前层级的所有元素出队
		queue = queue[length:]
	}
	return result
}

/* 
=============================== 二叉树的层次遍历（自底向上） =======================
给定一个二叉树，返回其节点值自底向上的层次遍历。 （即按从叶子节点所在层到根节点所在的层，逐层从左向右遍历）
例如：
给定二叉树 [3,9,20,null,null,15,7],

    3
   / \
  9  20
    /  \
   15   7

返回其自底向上的层次遍历为：

[
  [15,7],
  [9,20],
  [3]
]

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/binary-tree-level-order-traversal-ii
*/
func levelOrderBottom(root *TreeNode) [][]int {
	// BFS
	result := levelOrder(root)
	// reverse
	for i, j := 0, len(result) - 1 ; i < j; i, j = i + 1, j - 1 {
		result[i], result[j] = result[j], result[i]
	}
	return result
}

/* 
=================== 二叉树的锯齿形层次遍历 ====================
给定一个二叉树，返回其节点值的锯齿形层次遍历。（即先从左往右，再从右往左进行下一层遍历，以此类推，层与层之间交替进行）。
例如：
给定二叉树 [3,9,20,null,null,15,7],

    3
   / \
  9  20
    /  \
   15   7

返回锯齿形层次遍历如下：

[
  [3],
  [20,9],
  [15,7]
]

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/binary-tree-zigzag-level-order-traversal
*/
func zigzagLevelOrder(root *TreeNode) [][]int {
	result := make([][]int, 0)
	if root == nil {
		return result
	}
	queue := []*TreeNode{root}
	reverse := false
	for len(queue) > 0 {
		length := len(queue)
		level := make([]int, 0, length)
		for _, node := range queue {
			level = append(level, node.Val)
			if node.Left != nil {
				queue = append(queue, node.Left)
			}
			if node.Right != nil {
				queue = append(queue, node.Right)
			}
		}
		// 是否需要翻转
		if reverse {
			for i := 0; i < len(level) / 2; i ++ {
				level[i], level[len(level) - 1 - i] = level[len(level) - 1 - i], level[i]
			}
		}
		reverse = !reverse
		result = append(result, level)
		queue = queue[length:]
	}
	return result
}

/* 
====================  验证二叉搜索树 ==============================

给定一个二叉树，判断其是否是一个有效的二叉搜索树。
假设一个二叉搜索树具有如下特征：
    节点的左子树只包含小于当前节点的数。
    节点的右子树只包含大于当前节点的数。
    所有左子树和右子树自身必须也是二叉搜索树。

示例 1:
输入:
    2
   / \
  1   3
输出: true

示例 2:
输入:
    5
   / \
  1   4
     / \
    3   6
输出: false
解释: 输入为: [5,1,4,null,null,3,6]。
     根节点的值为 5 ，但是其右子节点值为 4 。

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/validate-binary-search-tree
*/
// 思路1：利用中序遍历，检查结果列表是否有序
/* func isValidBST(root *TreeNode) bool {
	list := inOrderTraversal(root)
	fmt.Println("list:", list)
	for i := 0; i < len(list) - 1; i ++ {
		if list[i] >= list[i + 1] {
			return false
		}
	}
	return true
}
func inOrderTraversal(root *TreeNode) []int {
	// nil && leaf
	if root == nil {
		return []int{}
	}

	// divide
	left := inOrderTraversal(root.Left)
	right := inOrderTraversal(root.Right)

	// conquer
	list := make([]int, 0, len(left) + len(right) + 1)
	list = append(list, left...)
	list = append(list, root.Val)
	list = append(list, right...)
	return list
} */

// 中序遍历（循环）
/* func isValidBST(root *TreeNode) bool {
	if root == nil {
		return true
	}
	stack := make([]*TreeNode, 0) 
	inOrder := -1 << 63
	for root != nil || len(stack) > 0 {
		for root != nil {
			stack = append(stack, root)
			root = root.Left
		}
		node := stack[len(stack) - 1]
		if node.Val <= inOrder {
			return false
		}
		stack = stack[:len(stack) - 1]
		inOrder = node.Val
		root = node.Right
	}
	return true
} */

// 思路 2：分治法，判断左 MAX < 根 < 右 MIN
/* func isValidBST(root *TreeNode) bool {
	r := BST(root)
	return r.isValidBST
}
type R struct {
	isValidBST bool
	max, min *TreeNode
}
func BST(root *TreeNode) R {
	result := R{}
	if root == nil {
		result.isValidBST = true
		return result
	}

	left := BST(root.Left)
	right := BST(root.Right)

	if !left.isValidBST || !right.isValidBST {
		result.isValidBST = false
		return result
	}

	if left.max != nil && left.max.Val >= root.Val {
		result.isValidBST = false
		return result
	}

	if right.min != nil && right.min.Val <= root.Val {
		result.isValidBST = false
		return result
	}

	result.isValidBST = true
    // 如果左边还有更小的3，就用更小的节点，不用4
    //  5
    // / \
    // 1   4
    //      / \
	//     3   6
	result.min = root
	if left.min != nil {
		result.min = left.min
	}

	result.max = root
	if right.max != nil {
		result.max = right.max
	}
	return result
} */

// 思路3：二叉搜索树对于每一个节点都有 left<root<right，即对于每一个节点都有 l<v<r，v∈(l,r)
// 故可设计一个函数f(root, lower, upper)，对left设置上限为root，对right设置下限为root，
// 初始lower=int64.Min upper=int64.max，递归执行，对不符合v∈(l,r)条件的直接返回 false
func isValidBST(root *TreeNode) bool {
	return isBST(root, -1 << 63, 1 << 63 - 1)
}
func isBST(root *TreeNode, lower, upper int) bool {
	if root == nil {
		return true
	}
	if lower >= root.Val || upper <= root.Val {
		return false
	}
	left := isBST(root.Left, lower, root.Val)
	right := isBST(root.Right, root.Val, upper)
	return left && right
}

/* 
给定二叉搜索树（BST）的根节点和要插入树中的值，将值插入二叉搜索树。 返回插入后二叉搜索树的根节点。 输入数据保证，新值和原始二叉搜索树中的任意节点值都不同。
注意，可能存在多种有效的插入方式，只要树在插入后仍保持为二叉搜索树即可。 你可以返回任意有效的结果。
例如, 
给定二叉搜索树:
        4
       / \
      2   7
     / \
    1   3
和 插入的值: 5
你可以返回这个二叉搜索树:
         4
       /   \
      2     7
     / \   /
    1   3 5
或者这个树也是有效的:
         5
       /   \
      2     7
     / \   
    1   3
         \
          4
提示：

    给定的树上的节点数介于 0 和 10^4 之间
    每个节点都有一个唯一整数值，取值范围从 0 到 10^8
    -10^8 <= val <= 10^8
    新值和原始二叉搜索树中的任意节点值都不同

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/insert-into-a-binary-search-tree
*/
/* func insertIntoBST(root *TreeNode, val int) *TreeNode {
	node := new(TreeNode)
	node.Val = val
	// root 为空，则用val给node创建一个节点并返回
	if root == nil {
		root = node
		return root
	}

	// 用一个指针指向root，保持root指向树的根节点
	r := root
	// 用一个指针指向当前节点的父节点
	parent := root
	for r != nil {
		// 记录父节点
		parent = r
		// 深入左、右节点
		if val < r.Val {
			r = r.Left
		} else {
			r = r.Right
		}
	}
	// 出循环的时候必定是抵达了val该插入的位置
	// 此时再通过该位置的父节点来定位插入即可
	if val < parent.Val {
		parent.Left = node
	}
	if val > parent.Val {
		parent.Right = node
	}
	return root
} */

// DFS查找插入位置
func insertIntoBST(root *TreeNode, val int) *TreeNode {
	if root == nil {
		root = &TreeNode{Val : val}
		return root
	}
	if val < root.Val {
		root.Left = insertIntoBST(root.Left, val)
	} else {
		root.Right = insertIntoBST(root.Right, val)
	}
	return root
}








// ============================ 案列测试 ==============================
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
	node2.Left = node5

	return root
}

// 先序遍历（递归DFS-从上到下）
// 输出：[0 1 3 4 2 5]
func preorderTraversalDFSTest() {
	root := buildTree()
	fmt.Println(preorderTraversalDFS(root))
}

// 归并排序
// 输出：[0 1 2 2 3 4 5 6 6 7 8 9 11]
func mergeSortTest() {
	arr := []int{1,5,2,4,6,8,6,7,9,2,3,0,11}
	fmt.Println(MergeSort(arr))
}

// 快速排序
// 输出：[0 1 2 2 3 4 5 6 6 7 8 9 11]
func QuickSortTest() {
	arr := []int{1,5,2,4,6,8,6,7,9,2,3,0,11}
	fmt.Println("qs1:", QuickSort(arr))

	arr2 := []int{1,5,2,4,6,8,6,7,9,2,3,0,11}
	fmt.Println("qs2:", QuickSort2(arr2))
}

// 二叉树最大深度
// 输出：3
func maxDepthTest() {
	root := buildTree()
	fmt.Println(maxDepth(root))
}

// 二叉平衡树
// 输出：true
func isBalancedTest() {
	root := buildTree()
	fmt.Println(isBalanced(root))
}

// 二叉树中的最大路径和
// 输出：12
func maxPathSumTest() {
	root := buildTree()
	fmt.Println(maxPathSum(root))
}

// 二叉树的最近公共祖先
// 输出：0
func lowestCommonAncestorTest() {
	root := buildTree()
	p := root.Left.Right
	q := root.Right.Left
	fmt.Println(lowestCommonAncestor(root, p, q).Val)
}

// 二叉树层级遍历
// 输出：[[0] [1 2] [3 4 5]]
func levelOrderTest() {
	root := buildTree()
	fmt.Println(levelOrder(root))
}

// 自底向上遍历二叉树层级
// 输出：[[3 4 5] [1 2] [0]]
func levelOrderBottomTest() {
	root := buildTree()
	fmt.Println(levelOrderBottom(root))
}

// 二叉树的锯齿形层次遍历
// 输出：[[0] [2 1] [3 4 5]]
func zigzagLevelOrderTest() {
	root := buildTree()
	fmt.Println(zigzagLevelOrder(root))
}

// 验证二叉搜索树
// 输出：
func isValidBSTTest() {
	root := buildTree()
	fmt.Println(isValidBST(root))
}

// 
func insertIntoBSTTest() {
	var root = new(TreeNode)
	fmt.Println(insertIntoBST(root, 1))
}

func main() {
	// 分治法二叉树先序遍历
	// preorderTraversalDFSTest()

	// 分治法归并排序
	// mergeSortTest()

	// 分治法快速排序
	// QuickSortTest()

	// maxDepthTest()
	// isBalancedTest()
	// maxPathSumTest()
	// lowestCommonAncestorTest()
	// levelOrderTest()
	// levelOrderBottomTest()
	// zigzagLevelOrderTest()
	// isValidBSTTest()
	insertIntoBSTTest()
}