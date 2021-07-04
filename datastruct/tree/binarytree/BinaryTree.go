package binarytree

/* 
	二叉树
*/

/* 
	二叉树遍历：
		1、先（前）序遍历
			首先访问根节点，然后遍历左子树，最后遍历右子树。
		2、中序遍历
			先遍历左子树，然后访问根节点，最后遍历右子树。
		3、后续遍历
			先遍历左子树，然后遍历右子树，最后访问树的根节点。
	注：前、中、后指的是根节点的访问顺序，对于子树来说，都是先访问左子树，然后访问右子树。
*/

/* 
========================== 1、二叉树的前序遍历 ==========================
给你二叉树的根节点 root ，返回它节点值的 前序 遍历。
示例 1：
输入：root = [1,null,2,3]
输出：[1,2,3]

示例 2：
输入：root = []
输出：[]

示例 3：
输入：root = [1]
输出：[1]

示例 4：
输入：root = [1,2]
输出：[1,2]

示例 5：
输入：root = [1,null,2]
输出：[1,2]

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/data-structure-binary-tree/xeywh5/
*/
/* 
	方法一：递归-DFS
	思路：
		自顶向下深度优先递归遍历二叉树
	时间复杂度：O(n)
		n 是二叉树节点个数
	空间复杂度：O(n)
*/
func preorderTraversal(root *TreeNode) []int {
	if root == nil {
		return []int{}
	}
	var DFS func(root *TreeNode, result *[]int)
	DFS = func(root *TreeNode, result *[]int) {
		if root == nil {
			return
		}
		*result = append(*result, root.Val)
		DFS(root.Left, result)
		DFS(root.Right, result)
	}
	result := make([]int, 0)
	DFS(root, &result)
	return result
}

/* 
	方法二：循环-DFS
	思路：
		采用栈来代替递归，实现循环遍历二叉树
	时间复杂度：O(n)
		n 是二叉树节点个数
	空间复杂度：O(n)
*/
func preorderTraversal(root *TreeNode) []int {
	if root == nil {
		return []int{}
	}
	result := make([]int, 0)
	stack := make([]*TreeNode, 0)
	for root != nil || len(stack) > 0 {
		for root != nil {
			result = append(result, root.Val)
			// 入栈
			stack = append(stack, root)
			root = root.Left			
		}
		// 出栈
		node := stack[len(stack) - 1]
		stack = stack[:len(stack) -1]
		root = node.Right
	}
	return result
}

/* 
	方法三：分治法
	思路：
		利用分治法自底向上合并结果返回
	时间复杂度：O(n)
		n 是二叉树节点个数
	空间复杂度：O(n)

	DFS 深度搜索（从上到下） 和分治法（从下到上）区别：
		前者一般将最终结果通过指针参数传入，后者一般递归返回结果最后合并
*/
func preorderTraversal(root *TreeNode) []int {
	if root == nil {
		return []int{}
	}

	var DAC func(root *TreeNode) []int
	DAC = func(root *TreeNode) []int {
		result := make([]int, 0)
		
		// 返回条件（null & leaf） leaf：叶子
		if root == nil {
			return result
		}

		// 分治（Divide）
		left := DAC(root.Left)
		right := DAC(root.Right)

		// 合并结果（Conquer）
		result = append(result, root.Val)
		result = append(result, left...)
		result = append(result, right...)
		return result
	}
	return DAC(root)
}

/* 
	方法四：Morris 遍历
	思路：
			根据 Morris 算法，利用二叉树中的空闲指针来完成后序遍历，具体如下：
			1、假设当前遍历到的节点是 x；
			2、如果 x 无左子树（左子树为空），将 x 的值加入答案，并遍历 x 的右子树，
				即 x=x.Right；
			3、如果 x 有左子树，在 x 的左子树中找到 x 在中序遍历下的前驱节点 preX，
				即 x 的左子树中的最右的节点，然后做以下判断：
				（1）如果 preX 无右子树（右子树为空），将 preX 的右指针指向当前节点 x，
					即 preX.Right=x，将 x 的前驱节点 preX 与 x 相连，以便后续处理 x。
					然后将 x 的值加入结果集，然后处理 x 的左子节点，即 x=x.Left。
				（2）如果 preX 有右子树，此时说明 x 及其左子树都已处理完成，
					需要将断开 preX 与 x 的连接，即 preX.Right = nil。往下处理
					x 的右子树，即 x=x.Right。
			4、重复步骤 2 和步骤 3，直到遍历结束。
	时间复杂度：O(n)
		其中 n 是二叉树的节点数。没有左子树的节点只被访问一次，有左子树的
		节点被访问两次。
	空间复杂度：O(1)
		只操作已经存在的指针（树的空闲指针），因此只需要常数的额外空间。
*/
func preorderTraversal(root *TreeNode) []int {
	result := make([]int, 0)
	if root == nil {
		return result
	}
	for root != nil {
		// 左子树为空，则将当前节点的值存入结果集，然后处理右子树
		if root.Left == nil {
			result = append(result, root.Val)
			root = root.Right
			continue
		}
		// x 有左子树，从 x 的左子树中找到 x 在中序遍历下的前驱节点 preX
		preRoot := root.Left
		for preRoot.Right != nil && preRoot.Right != root {
			preRoot = preRoot.Right
		}
		// 前驱节点 preX 的右子树为空，则将 x 的值存入结果集，然后
		// 再将 preX 与 x 相连，以便后续处理 x，接着处理 x 的左子树
		if preRoot.Right == nil {
			result = append(result, root.Val)
			preRoot.Right = root
			root = root.Left
			continue
		} 
		// 前驱节点 preX 的右子树不为空，说明此时 preX == x，
		// x 及其的左子树已经处理完成，此时需要断开 preX 与 x 的连接，
		// 处理 x 的右子树
		preRoot = nil
		root = root.Right
	}
	return result
}

/* 
========================== 2、二叉树的中序遍历 ==========================
给定一个二叉树的根节点 root ，返回它的 中序 遍历。

示例 1：
输入：root = [1,null,2,3]
输出：[1,3,2]

示例 2：
输入：root = []
输出：[]

示例 3：
输入：root = [1]
输出：[1]

示例 4：
输入：root = [1,2]
输出：[2,1]

示例 5：
输入：root = [1,null,2]
输出：[1,2]

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/data-structure-binary-tree/xecaj6/
*/
/* 
	方法一：递归DFS
	思路：
		自顶向下深度优先递归遍历
	时间复杂度：O(n)
	空间复杂度：O(n)
*/
func inorderTraversal(root *TreeNode) []int {
	result := make([]int, 0)
	if root == nil {
		return result
	}

	var DFS func(root *TreeNode, result *[]int)
	DFS = func(root *TreeNode, result *[]int) {
		if root == nil {
			return
		}

		DFS(root.Left, result)
		*result = append(*result, root.Val)
		DFS(root.Right, result)
	}

	DFS(root, &result)
	return result
}

/* 
	方法二：循环DFS
	思路：
		使用栈代替递归来实现循环DFS遍历
	时间复杂度：O(n)
	空间复杂度：O(n)
*/
func inorderTraversal(root *TreeNode) []int {
	result := make([]int, 0)
	if root == nil {
		return result
	}
	stack := make([]*TreeNode, 0)
	for root != nil || len(stack) > 0 {
		for root != nil {
			// push
			stack = append(stack, root)
			root = root.Left
		}
		// pop
		node := stack[len(stack) - 1]
		stack = stack[:len(stack) - 1]

		result = append(result, node.Val)
		root = node.Right
	}
	return result
}

/* 
	方法三：分治法
	思路：
		使用分治自底向上合并结果
	时间复杂度：O(n)
	空间复杂度：O(n)
*/
func inorderTraversal(root *TreeNode) []int {
	if root == nil {
		return []int{}
	}

	var DAC func(root *TreeNode) []int
	DAC = func(root *TreeNode) []int {
		result := make([]int, 0)
		// leaf & nil
		if root == nil {
			return result
		}

		// 分治（Divide）
		left := DAC(root.Left)
		right := DAC(root.Right)

		// 合并（Conquer）
		result = append(result, left...)
		result = append(result, root.Val)
		result = append(result, right...)
		return result
	}
	return DAC(root)
}

/* 
	方法三：Morris 中序遍历
	思路：
		根据 Morris 算法，利用二叉树中的空闲指针来完成后序遍历，具体如下：
		1、假设当前遍历到的节点为 x。
		2、如果 x 无左子树，将 x 的值存入结果集，接着处理 x 的右子树，
				即 x=x.Right。
		3、如果 x 有左子树，则在 x 的左子树中找到 x 的前驱节点 preX
			即 x 的左子树中的最右的节点，然后做以下判断：
			（1）如果 preX 无右子树（右子树为空），则将其右指针指向 x，即
				将 x 的前驱节点 preX 与 x 相连，以便后续处理 x，然后处理
				x 的左子树，即 x=x.Left。
			（2）如果 preX 有右子树，说明 preX.Right = x，即 x 的
				左子树已处理完成（注：此处与先序遍历不同，在中序遍历中，x 的左子树处理完后，
				x 还没被处理），此时将 x 的值存入结果集，然后需要断开 preX 与 x 的连接，
				即 preX.Right = nil，再接着处理 x 的右子树，即 x=x.Right。
		4、重复上述操作，直至访问完整棵树。
	时间复杂度：O(n)
		其中 n 是二叉树的节点数。没有左子树的节点只被访问一次，有左子树的
		节点被访问两次。
	空间复杂度：O(1)
		只操作已经存在的指针（树的空闲指针），因此只需要常数的额外空间。
*/
func inorderTraversal(root *TreeNode) []int {
	result := make([]int, 0)
	if root == nil {
		return result
	}
	for root != nil {
		// x 无左子树，将 x 的值存入结果集，接着处理 x 的右子树
		if root.Left == nil {
			result = append(result, root.Val)
			root = root.Right
			continue
		}

		// x 有左子树，从 x 的左子树中找到 x 在中序遍历下的前驱节点 preX
		preRoot := root.Left
		for preRoot.Right != nil && preRoot.Right != root {
			preRoot = preRoot.Right
		}

		// preX 无右子树，连接 preX 和 x，处理 x 的左子树
		// 注：此处与先序遍历不同，先处理完 x 的左子树才能处理 x
		if preRoot.Right == nil {
			preRoot.Right = root
			root = root.Left
			continue
		}

		// preX 有右子树，此时 preX = x，说明 x 的左子树已经处理完成，
		// 需要断开 preX 与 x 的连接，接着处理 x 及其右子树
		preRoot.Right = nil
		result = append(result, root.Val)
		root = root.Right
	}
	return result
}

/* 
========================== 3、二叉树的后序遍历 ==========================
给定一个二叉树，返回它的 后序 遍历。

示例:
输入: [1,null,2,3]  
   1
    \
     2
    /
   3 

输出: [3,2,1]
进阶: 递归算法很简单，你可以通过迭代算法完成吗？

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/data-structure-binary-tree/xebrb2/
*/
/* 
	方法一：递归DFS
	思路：
		自顶向下DFS遍历
	时间复杂度：O(n)
	空间复杂度：O(n)
*/
func postorderTraversal(root *TreeNode) []int {
	result := make([]int, 0)
	if root == nil {
		return result
	}
	var DFS func(root *TreeNode, result *[]int) 
	DFS = func(root *TreeNode, result *[]int) {
		if root == nil {
			return
		}
		DFS(root.Left, result)
		DFS(root.Right, result)
		*result = append(*result, root.Val)
	}
	DFS(root, &result)
	return result
}

/* 
	方法二：循环DFS
	思路：
		借助栈来代替递归实现循环DFS
		核心：根节点必须在右节点弹出之后，再弹出
	时间复杂度：O(n)
	空间复杂度：O(n)
*/
func postorderTraversal(root *TreeNode) []int {
	result := make([]int, 0)
	if root == nil {
		return result
	}
	stack := make([]*TreeNode, 0)
	// 标记最后弹出的节点
	var lastVisit *TreeNode
	for root != nil || len(stack) > 0 {
		for root != nil {
			// push
			stack = append(stack, root)
			// 深入左节点
			root = root.Left
		}
		// 这里只是查看并不急着弹出，相当于 peek
		node := stack[len(stack) - 1]
		// 根节点必须必须在右节点弹出之后再弹出
		if node.Right == nil || node.Right == lastVisit {
			// pop
			result = append(result, node.Val)
			stack = stack[:len(stack) - 1]
			// 标记当前这个节点已经弹出过
			lastVisit = node
		} else {
			// 深入右节点
			root = node.Right
		}
	}
	return result
}

/* 
	方法三：分治法
	思路：
		使用分治法自底向上合并结果
	时间复杂度：O(n)
	空间复杂度：O(n)
*/
func postorderTraversal(root *TreeNode) []int {
	result := make([]int, 0)
	if root == nil {
		return result
	}

	var DAC func(root *TreeNode) []int
	DAC = func(root *TreeNode) []int {
		result := make([]int, 0)
		// leaf & nil
		if root == nil {
			return result
		}

		// Divide（分治）
		left := DAC(root.Left)
		right := DAC(root.Right)

		// Conquer（合并）
		result = append(result, left...)
		result = append(result, right...)
		result = append(result, root.Val)
		return result
	}
	return DAC(root)
}

/* 
	方法四：Morris 后序遍历
	思路：
		根据 Morris 算法，利用二叉树中的空闲指针来完成后序遍历，具体如下：
		1、假设当前遍历到的节点是 x。
		2、如果 x 无左子树，处理 x 的右子树，即 x=x.Rihgt。
		3、如果 x 有左子树，在 x 的左子树中寻找 x 在中序遍历下的前驱节点 preX，
			即在x 的左子树中找到最右的节点，做以下判断：
			（1）preX 无右子树，用 preX 的右指针将 x 与 x 的前驱节点 preX 相连，
				即 preX.Right = x，然后处理 x 的左子树。
			（2）preX 有右子树，此时 preX.Right = x，说明 x 的左子树已经处理完成，
				此时需要断开 preX 与 x 的连接，即 pre.Right = nil，接着倒序输出从
				x 的左子节点到 preX 这条路径上的所有节点，即 
				最后处理 x 的右子树，即 x = x.Right
		4、重复步骤 2 和步骤 3，直到遍历结束。
*/
// 反转
func reverse(a []int) {
    for i, n := 0, len(a); i < n/2; i++ {
        a[i], a[n-1-i] = a[n-1-i], a[i]
    }
}
func postorderTraversal(root *TreeNode) (res []int) {
    addPath := func(node *TreeNode) {
        resSize := len(res)
        for ; node != nil; node = node.Right {
            res = append(res, node.Val)
        }
        reverse(res[resSize:])
    }

	x := root
    for x != nil {
		// x 无左子树，处理 x 的右子树
		if x.Left == nil {
			x = x.Right
			continue
		}

		// x 有左子树，从 x 的左子树中找到 x 在中序遍历下的前驱节点 preX
		preX := x.Left
		for preX.Right != nil && preX.Right != x {
			preX = preX.Right
		}

		// preX 无右子树，连接 preX 和 x，处理 x 的左子树
		if preX.Right == nil {
			preX.Right = x
			x = x.Left
			continue
		}

		// preX 有右子树，此时 preX.Right = x，说明 x 的左子树都已经处理完成
		// 此时需要断开 preX 与 x 的连接，然后倒序输出 x 的左节点到 preX 这条
		// 路径下的所有节点，最后处理 x 的右子树
		preX.Right = nil
		addPath(x.Left)
		x = x.Right
	}
	// 后序遍历最后处理根节点
    addPath(root)
    return
}

/* 
========================== 4、二叉树的层序遍历 ==========================
给你一个二叉树，请你返回其按 层序遍历 得到的节点值。
（即逐层地，从左到右访问所有节点）。

示例：
二叉树：[3,9,20,null,null,15,7],
    3
   / \
  9  20
    /  \
   15   7

返回其层序遍历结果：
[
  [3],
  [9,20],
  [15,7]
]

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/data-structure-binary-tree/xefh1i/
*/
/* 
	方法一：循环BFS
	思路：
		借助于队列，对二叉树采用广度优先搜索（BFS），用当前层级的元素去推导下一
		层级的元素。
	时间复杂度：O(n)
		n 是二叉树元素的个数
	空间复杂度：O(n)
*/
func levelOrder(root *TreeNode) [][]int {
	result := make([][]int, 0)
	if root == nil {
		return result
	}
	// 初始化层级队列
	queue := []*TreeNode{root}
	for len(queue) > 0 {
		// 处理当前层级的元素
		n := len(queue)
		list := make([]int, n)
		for i, v := range queue {
			list[i] = v.Val
			// 装入下一层级的元素
			if v.Left != nil {
				queue = append(queue, v.Left)
			}
			if v.Right != nil {
				queue = append(queue, v.Right)
			}
		}
		// 添加当前层级的结果到结果集
		result = append(result, list)
		// 移除已经处理的元素
		queue = queue[n:]
	}
	return result
}