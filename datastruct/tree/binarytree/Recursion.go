package binarytree

/* 
	使用递归思想解决二叉树问题
		1、自顶向下
			意味着在每个递归层级，我们将首先访问节点来计算一些值，并在递归
			调用函数时将这些值传递到子节点。 所以 “自顶向下” 的解决方案可以
			被认为是一种前序遍历。
		2、自底向上
			在每个递归层次上，我们首先对所有子节点递归地调用函数，然后根据返回
			值和根节点本身的值得到答案。 这个过程可以看作是后序遍历的一种。
	小结：
		1、“自顶向下”的使用场景：
			（1）根据当前传入的参数，能够得出当前节点的答案。
			（2）根据这些参数和节点本身的值，可以决定子节点的参数传递。
		2、“自底向上”使用场景：
			对于树中的任意一个节点，如果知道它子节点的答案，
			就能计算出该节点的答案。
*/

/* 
========================== 1、二叉树的最大深度 ==========================
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

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/data-structure-binary-tree/xoh1zg/
 */
/* 
	方法一：递归DFS（自顶向下）
	思路：
		定义好当前深度 depth 和最大深度 maxDph，然后将它们当做参数传递
		给递归函数，在每一个节点中计算 depth 的值，并对比 depth 和 maxDph
		的值来更新 maxDph。
		注意：maxDph 要以引用方式进行传递。
	时间复杂度：O(n)
		n 是二叉树节点个数，我们需要在每一个节点中都对 maxDph 进行计算。
	空间复杂度：O(1)
*/
func maxDepth(root *TreeNode) int {
	maxDph := 0
	if root == nil {
		return maxDph
	}
	var DFS func(root *TreeNode, depath int, maxDph *int)
	DFS = func(root *TreeNode, depath int, maxDph *int) {
		if root == nil {
			return
		}
		// 当前节点深度
		depath += 1
		// 更新最大深度
		if depath > *maxDph {
			*maxDph = depath
		}
		DFS(root.Left, depath, maxDph)
		DFS(root.Right, depath, maxDph)
	}
	DFS(root, 0, &maxDph)
	return maxDph
}
/* 
	方法二：分治法（自底向上）
	思路：
		采用分治法，自底向上合并结果
	时间复杂度：O(n)
		n 是二叉树节点个数，我们需要在每一个节点中都对 maxDph 进行计算。
	空间复杂度：O(1)
*/
func maxDepth(root *TreeNode) int {
	maxDph := 0
	if root == nil {
		return maxDph
	}
	var DAC func(root *TreeNode) int
	DAC = func(root *TreeNode) int {
		if root == nil {
			return 0
		}
		left := DAC(root.Left)
		right := DAC(root.Right)
		depath := max(left, right) + 1
		return depath
	}
	return DAC(root)
}
func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

/* 
	方法三：BFS（层级遍历）
	思路：
		采用广度优先算法（BFS）获取树的层级，由树的层级便可以得出树的最大深度。
	时间复杂度：O(n)
		n 是二叉树节点个数，我们需要在每一个节点中都扩展下一个层级的元素。
	空间复杂度：O(1)
*/
func maxDepth(root *TreeNode) int {
	maxDph := 0
	if root == nil {
		return maxDph
	}
	queue := []*TreeNode{root}
	for len(queue) > 0 {
		n := len(queue)
		for _, v := range queue {
			if v.Left != nil {
				queue = append(queue, v.Left)
			}
			if v.Right != nil {
				queue = append(queue, v.Right)
			}
		}
		// 处理完当前层级，深度 +1
		maxDph ++
		queue = queue[n:]
	}
	return maxDph
}

/* 
========================== 2、对称二叉树 ==========================
给定一个二叉树，检查它是否是镜像对称的。

例如，二叉树 [1,2,2,3,4,4,3] 是对称的。
    1
   / \
  2   2
 / \ / \
3  4 4  3

但是下面这个 [1,2,2,null,3,null,3] 则不是镜像对称的:
    1
   / \
  2   2
   \   \
   3    3

进阶：
你可以运用递归和迭代两种方法解决这个问题吗？

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/data-structure-binary-tree/xoxzgv/
*/
/* 
	方法一：递归
	思路：
		如果一个树的左子树与右子树镜像对称，那么这个树是对称的。
		因此，该问题可以转化为：两个树在什么情况下互为镜像？
		如果同时满足下面的条件，两个树互为镜像：
			1、它们的两个根结点具有相同的值
			2、每个树的右子树都与另一个树的左子树镜像对称
		我们可以实现这样一个递归函数，通过「同步移动」两个指针的方法来遍历这棵树，
		p 指针和 q 指针一开始都指向这棵树的根，随后 p 右移时，q 左移，p 左移时，q 右移。
		每次检查当前 p 和 q 节点的值是否相等，如果相等再判断左右子树是否对称。
	时间复杂度：O(n)
		这里遍历了这棵树，渐进时间复杂度为 O(n)。
	空间复杂度：O(n)
		这里的空间复杂度和递归使用的栈空间有关，这里递归层数不超过 n，故渐进空间复杂度为 O(n)。
*/
func isSymmetric(root *TreeNode) bool {
	return check(root, root)
}
func check(p, q *TreeNode) bool {
	if p == nil && q == nil {
		return true
	}
	// 只有一个为 nil，不对称
	if p == nil || q == nil {
		return false
	}
	return p.Val == q.Val && check(p.Left, q.Right) && check(p.Right, q.Left)
}

/* 
	方法二：迭代
	思路：
		首先我们引入一个队列，这是把递归程序改写成迭代程序的常用方法。
		初始化时我们把根节点入队两次。每次提取两个结点并比较它们的值
		（队列中每两个连续的结点应该是相等的，而且它们的子树互为镜像），
		然后将两个结点的左右子结点按相反的顺序插入队列中。当队列为空时，
		或者我们检测到树不对称（即从队列中取出两个不相等的连续结点）时，
		该算法结束。
	时间复杂度：O(n)
		这里遍历了这棵树，渐进时间复杂度为 O(n)。
	空间复杂度：O(n)
		这里的空间复杂度和递归使用的栈空间有关，这里递归层数不超过 n，故渐进空间复杂度为 O(n)。
*/
func isSymmetric(root *TreeNode) bool {
	u, v := root, root
	queue := []*TreeNode{}
	queue = append(queue, u)
	queue = append(queue, v)

	for len(queue) > 0 {
		u, v = queue[0], queue[1]
		// 删除已取出的的两个元素
		queue = queue[2:]
		if u == nil && v == nil {
			continue
		}
		// 只有一个为 nil，不对称
		if u == nil || v == nil {
			return false
		}
		if u.Val != v.Val {
			return false
		}
		queue = append(queue, u.Left)
		queue = append(queue, v.Right)

		queue = append(queue, u.Right)
		queue = append(queue, v.Left)
	}
	return true
}

/* 
========================== 3、路径总和 ==========================
给你二叉树的根节点 root 和一个表示目标和的整数 targetSum ，判断该树中是否存在 
根节点到叶子节点 的路径，这条路径上所有节点值相加等于目标和 targetSum 。

叶子节点 是指没有子节点的节点。

示例 1：
		 5
		/ \
	   4   8
	  /	  / \
	 11	 13  4
	/ \		  \
   7   2	   1

输入：root = [5,4,8,11,null,13,4,7,2,null,null,null,1], targetSum = 22
输出：true

示例 2：
	1
   / \
  2   3

输入：root = [1,2,3], targetSum = 5
输出：false

示例 3：
	1
   /
  2
输入：root = [1,2], targetSum = 0
输出：false

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/data-structure-binary-tree/xo566j/
*/
/* 
	方法一：递归DFS
	思路：
		携带目标值，深度优先遍历二叉树，每到一个节点，就用目标值减去当前节点的值，
		当到达叶子节点时，目标值刚好为0，则说明存在这样的路径。
	时间复杂度：O(N)
		其中 N 是树的节点数。对每个节点访问一次。
	空间复杂度：O(H)
		其中 H 是树的高度。空间复杂度主要取决于递归时栈空间的开销，
		最坏情况下，树呈现链状，空间复杂度为 O(N)。
*/
func hasPathSum(root *TreeNode, targetSum int) bool {
	result := false
	var DFS func(root *TreeNode, targetSum int, result *bool) 
	DFS = func(root *TreeNode, targetSum int, result *bool) {
		if root == nil {
			return
		}
		targetSum -= root.Val
		// 是叶子节点，且目标值为 0
		if targetSum == 0 && root.Left == nil && root.Right == nil {
			*result = true
		}
		DFS(root.Left, targetSum, result)
		DFS(root.Right, targetSum, result)
	}
	DFS(root, targetSum, &result)
	return result
}
/* 
	方法一：递归DFS - 优化
	思路：
		观察要求我们完成的函数，我们可以归纳出它的功能：询问是否存在从当前节点 
		root 到叶子节点的路径，满足其路径和为 sum。
		假定从根节点到当前节点的值之和为 val，我们可以将这个大问题转化为一个小问题：
			是否存在从当前节点的子节点到叶子的路径，满足其路径和为 sum - val。
		不难发现这满足递归的性质，若当前节点就是叶子节点，那么我们直接判断 sum 
		是否等于 val 即可（因为路径和已经确定，就是当前节点的值，我们只需要判断该
		路径和是否满足条件）。若当前节点不是叶子节点，我们只需要递归地询问它的子节点
		是否能满足条件即可。
	时间复杂度：O(N)
		其中 N 是树的节点数。对每个节点访问一次。
	空间复杂度：O(H)
		其中 H 是树的高度。空间复杂度主要取决于递归时栈空间的开销，
		最坏情况下，树呈现链状，空间复杂度为 O(N)。平均情况下树的高度与节点数的对数
		正相关，空间复杂度为 O(log⁡N)。
*/
func hasPathSum(root *TreeNode, sum int) bool {
    if root == nil {
        return false
    }
    if root.Left == nil && root.Right == nil {
        return sum == root.Val
    }
    return hasPathSum(root.Left, sum - root.Val) || hasPathSum(root.Right, sum - root.Val)
}

/* 
	方法二：迭代
	思路：
		首先我们可以想到使用广度优先搜索的方式，记录从根节点到当前节点的路径和，
		以防止重复计算。这样我们使用两个队列，分别存储将要遍历的节点，以及根节点
		到这些节点的路径和即可。
	时间复杂度：O(N)
		其中 N 是树的节点数。对每个节点访问一次。
	空间复杂度：O(N)
		其中 N 是树的节点数。空间复杂度主要取决于队列的开销，
		队列中的元素个数不会超过树的节点数。
*/
func hasPathSum(root *TreeNode, sum int) bool {
    if root == nil {
        return false
    }
    queNode := []*TreeNode{root}
    queVal := []int{root.Val}
    for len(queNode) > 0 {
        now := queNode[0]
        queNode = queNode[1:]
        temp := queVal[0]
        queVal = queVal[1:]
        if now.Left == nil && now.Right == nil {
            if temp == sum {
                return true
            }
            continue
        }
        if now.Left != nil {
            queNode = append(queNode, now.Left)
            queVal = append(queVal, now.Left.Val + temp)
        }
        if now.Right != nil {
            queNode = append(queNode, now.Right)
            queVal = append(queVal, now.Right.Val + temp)
        }
    }
    return false
}