package main

import (
	"fmt"
)

/* 
================== 1、对称二叉树 ==================
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

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/symmetric-tree
*/
type TreeNode struct {
	Val int
	Left *TreeNode
	Right *TreeNode
}
/* 
	方法一：递归
	思路：
		如果一个树的左子树与右子树镜像对称，那么这个树是对称的。
		因此，该问题可以转化为：两个树在什么情况下互为镜像？
		如果同时满足下面的条件，两个树互为镜像：
			它们的两个根结点具有相同的值
			每个树的右子树都与另一个树的左子树镜像对称
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
func isSymmetric2(root *TreeNode) bool {
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
================== 2、路径总和 ==================
给定一个二叉树和一个目标和，判断该树中是否存在根节点到叶子节点的路径，
这条路径上所有节点值相加等于目标和。
说明: 叶子节点是指没有子节点的节点。

示例: 
给定如下二叉树，以及目标和 sum = 22，
              5
             / \
            4   8
           /   / \
          11  13  4
         /  \      \
        7    2      1
返回 true, 因为存在目标和为 22 的根节点到叶子节点的路径 5->4->11->2。

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/data-structure-binary-tree/xo566j/
*/
/* 
	方法一：递归
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
func hasPathSum2(root *TreeNode, sum int) bool {
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

/* 
================== 3、从中序与后序遍历序列构造二叉树 ==================
根据一棵树的中序遍历与后序遍历构造二叉树。
注意:
你可以假设树中没有重复的元素。

例如，给出
中序遍历 inorder = [9,3,15,20,7]
后序遍历 postorder = [9,15,7,20,3]

返回如下的二叉树：
    3
   / \
  9  20
    /  \
   15   7

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/construct-binary-tree-from-inorder-and-postorder-traversal
*/

/* 
	方法一：递归
	思路：
		首先解决这道题我们需要明确给定一棵二叉树，我们是如何对其进行中序遍历与后序遍历的：
    		中序遍历的顺序是每次遍历左孩子，再遍历根节点，最后遍历右孩子。
    		后序遍历的顺序是每次遍历左孩子，再遍历右孩子，最后遍历根节点。
		因此根据上文所述，我们可以发现后序遍历的数组最后一个元素代表的即为根节点。
		知道这个性质后，我们可以利用已知的根节点信息在中序遍历的数组中找到根节点所在
		的下标，然后根据其将中序遍历的数组分成左右两部分，左边部分即左子树，右边部分为
		右子树，针对每个部分可以用同样的方法继续递归下去构造。
	算法：
		为了高效查找根节点元素在中序遍历数组中的下标，我们选择创建哈希表来存储中序序列，
		即建立一个（元素，下标）键值对的哈希表。
		定义递归函数 helper(in_left, in_right) 表示当前递归到中序序列中当前子树的左右
		边界，递归入口为 helper(0, n - 1) ：
		如果 in_left > in_right，说明子树为空，返回空节点。
		选择后序遍历的最后一个节点作为根节点。
		利用哈希表 O(1) 查询当根节点在中序遍历中下标为 index。从 in_left 到 
		index - 1 属于左子树，从 index + 1 到 in_right 属于右子树。
		根据后序遍历逻辑，递归创建右子树 helper(index + 1, in_right) 和左子树 
		helper(in_left, index - 1)。
		
		注意这里有需要先创建右子树，再创建左子树的依赖关系。
		可以理解为在后序遍历的数组中整个数组是先存储左子树的节点，再存储右子树的节点，
		最后存储根节点，如果按每次选择「后序遍历的最后一个节点」为根节点，则先被构造出来
		的应该为右子树。
		返回根节点 root。
	时间复杂度：O(n)
		其中 n 是树中的节点个数。
	空间复杂度：O(n)
		我们需要使用 O(n) 的空间存储哈希表，以及 O(h)（其中 h 是树的高度）的空间表示
		递归时栈空间。这里 h<n，所以总空间复杂度为 O(n)。
*/
func buildTree(inorder []int, postorder []int) *TreeNode {
    idxMap := map[int]int{}
    for i, v := range inorder {
        idxMap[v] = i
    }
    var build func(int, int) *TreeNode
    build = func(inorderLeft, inorderRight int) *TreeNode {
        // 无剩余节点
        if inorderLeft > inorderRight {
            return nil
        }
		// 从后序数组中取根节点
        // 后序遍历的末尾元素即为当前子树的根节点
        val := postorder[len(postorder)-1]
        postorder = postorder[:len(postorder)-1]
        root := &TreeNode{Val: val}
        // 根据 val 在中序遍历的位置，将中序遍历划分成左右两颗子树
        // 由于我们每次都从后序遍历的末尾取元素，所以要先遍历右子树再遍历左子树
        inorderRootIndex := idxMap[val]
        root.Right = build(inorderRootIndex+1, inorderRight)
        root.Left = build(inorderLeft, inorderRootIndex-1)
        return root
    }
    return build(0, len(inorder)-1)
}

/* 
================== 4、从前序与中序遍历序列构造二叉树 ==================
根据一棵树的前序遍历与中序遍历构造二叉树。

注意:
你可以假设树中没有重复的元素。

例如，给出

前序遍历 preorder = [3,9,20,15,7]
中序遍历 inorder = [9,3,15,20,7]

返回如下的二叉树：

    3
   / \
  9  20
    /  \
   15   7

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/data-structure-binary-tree/xoei3r/
来源：力扣（LeetCode）
*/
/* 
	方法一：递归
	思路：
		与 从中序与后序遍历序列构造二叉树 一样，只是这次是从先序数组中找根节点，
		然后再从中序数组中找左右节点，而在构建分支子树的时候，先构建左子树，再构建
		右子树，最后返回 root 节点。
	时间复杂度：O(n)
		其中 n 是树中的节点个数。
	空间复杂度：O(n)
		我们需要使用 O(n) 的空间存储哈希表，以及 O(h)（其中 h 是树的高度）的空间表示
		递归时栈空间。这里 h<n，所以总空间复杂度为 O(n)。
*/
func buildTree2(preorder []int, inorder []int) *TreeNode {
	idxMap := make(map[int]int, len(inorder))
	for i, v := range inorder {
		idxMap[v] = i
	}
	var build func(int, int) *TreeNode
	build = func(inorderLeft, inorderRight int) *TreeNode {
		// 无剩余节点
		if inorderLeft > inorderRight {
			return nil
		}
		// 从先序数组中取根节点
		// 先序遍历的第一个元素即为当前子树的根节点
		value := preorder[0]
		preorder = preorder[1:]
		// 找到根节点在中序数组中的位置
		index := idxMap[value]
		// 根据 val 在中序遍历的位置，将中序遍历划分成左右两颗子树
		root := &TreeNode{Val:value}
		root.Left = build(inorderLeft, index - 1)
		root.Right = build(index + 1, inorderRight)
		return root
	}
	return build(0, len(preorder) - 1)
}

// ================== 案列测试 ==================
func myBuildTree() *TreeNode {
	n1 := &TreeNode{1, nil, nil}
	n2 := &TreeNode{2, nil, nil}
	n3 := &TreeNode{2, nil, nil}
	n4 := &TreeNode{3, nil, nil}
	n5 := &TreeNode{4, nil, nil}
	n6 := &TreeNode{3, nil, nil}
	n7 := &TreeNode{4, nil, nil}
	n1.Left = n2
	n1.Right = n3
	n2.Left = n4
	n2.Right = n5
	n3.Left = n7
	n3.Right = n6
	return n1
}

// ================== 1、对称二叉树 ==================
func isSymmetricTest() {
	root := myBuildTree()
	// res := isSymmetric(root)
	res := isSymmetric2(root)
	fmt.Println(res)
}

// ================== 2、路径总和 ==================
func hasPathSumTest() {
	root := myBuildTree()
	// res := hasPathSum(root, 8)
	res := hasPathSum2(root, 6)
	fmt.Println(res)
}

// ================== 3、从中序与后序遍历序列构造二叉树 ==================
func buildTreeTest() {
	inorder := []int{9,3,15,20,7}
	postorder := []int{9,15,7,20,3}
	res := buildTree(inorder, postorder)
	inOrderPrint(res)
}
// 树的中序遍历
func inOrderPrint(root *TreeNode) {
	if root == nil {
		return
	}
	fmt.Printf("%v ", root.Val)
	inOrderPrint(root.Left)
	inOrderPrint(root.Right)
}

// ================== 4、从前序与中序遍历序列构造二叉树 ==================
func buildTree2Test() {
	preorder := []int{3,9,20,15,7}
	inorder := []int{9,3,15,20,7}
	res := buildTree2(preorder, inorder)
	inOrderPrint(res)
}

func main() {
	// isSymmetricTest()
	// hasPathSumTest()
	// buildTreeTest()
	buildTree2Test()
}