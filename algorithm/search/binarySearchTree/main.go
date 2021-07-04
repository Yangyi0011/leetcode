package main

import (
	"fmt"
)

/* 
===================== 二分查找树 =====================
	定义：
	    每个节点中的值必须大于（或等于）存储在其左侧子树中的任何值。
    	每个节点中的值必须小于（或等于）存储在其右子树中的任何值。
*/
type TreeNode struct {
	Val int
	Left, Right *TreeNode
}

/* 
===================== 1、验证二叉搜索树 =====================
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

/* 
	方法一：DFS
	思路：
		根据二分查找树的性质：
			节点的左子树只包含小于当前节点的数。
			节点的右子树只包含大于当前节点的数。
			所有左子树和右子树自身必须也是二叉搜索树。
		我们可以定义每一颗子树的上限(upper)和下限(lower)，二叉查找树的每一
		个 node 节点的值必须处于该子树的上限与下限之间，否则不是二分查找树。
			对于每一个左子树的节点，其范围是：(lower, node.Val)
			对于每一个右子树的节点，其范围是：(node.Val, upper)
			根节点的上限和下限我们定义为：(int64.min, int64.max)
	时间复杂度：O(n)
		n 表示树的节点个数，我们需要完全验证每一个节点是否符合规则。
	空间复杂度：O(n)，
		其中 n 为二叉树的节点个数。递归函数在递归过程中需要为每一层递归函数
		分配栈空间，所以这里需要额外的空间且该空间取决于递归的深度，即二叉树的高度。
		最坏情况下二叉树为一条链，树的高度为 n ，递归最深达到 n 层，故最坏情况下空
		间复杂度为 O(n) 。
*/
func isValidBST(root *TreeNode) bool {
	return isBST(root, -1 << 63, 1 << 63 - 1)
}
func isBST(root *TreeNode, lower, upper int) bool {
	if root == nil {
		return true
	}
	if root.Val <= lower || root.Val >= upper {
		return false
	}
	left := isBST(root.Left, lower, root.Val)
	right := isBST(root.Right, root.Val, upper)
	return left && right
}

/* 
	方法二：中序遍历
	思路：
		根据二叉搜索树的性质：left < root < right 我们可知，对二叉搜索树
		进行中序遍历得出的结果集一定是有序的，无序的就不是二叉搜索树。
	时间复杂度：O(n)
		n 是树的节点个数，我们需要先进行中序遍历，再判断中序遍历的结果是否有序。
	空间复杂度：O(n)
		我们需要用长度为 n 的数组来存储中序遍历的结果。
*/
func isValidBST2(root *TreeNode) bool {
    result := make([]int, 0)
    inOrder(root, &result)
    // check order
    for i := 0; i < len(result) - 1; i++{
        if result[i] >= result[i+1] {
            return false
        }
    }
    return true
}
func inOrder(root *TreeNode, result *[]int)  {
    if root == nil {
        return
    }
    inOrder(root.Left, result)
    *result = append(*result, root.Val)
    inOrder(root.Right, result)
}

/* 
===================== 2、二叉搜索树中的插入操作 =====================
给定二叉搜索树（BST）的根节点和要插入树中的值，将值插入二叉搜索树。
返回插入后二叉搜索树的根节点。 输入数据 保证 ，新值和原始二叉搜索树
中的任意节点值都不同。
注意，可能存在多种有效的插入方式，只要树在插入后仍保持为二叉搜索树即可。 
你可以返回 任意有效的结果 。

示例 1：
输入：root = [4,2,7,1,3], val = 5
输出：[4,2,7,1,3,5]
解释：另一个满足题目要求可以通过的树是：

示例 2：
输入：root = [40,20,60,10,30,50,70], val = 25
输出：[40,20,60,10,30,50,70,null,null,25]

示例 3：
输入：root = [4,2,7,1,3,null,null,null,null,null,null], val = 5
输出：[4,2,7,1,3,5]

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/insert-into-a-binary-search-tree
*/

/* 
	方法一：DFS-自顶向下
	思路：
		从根节点往下，利用二叉查找树的性质查找合适的插入点，插入方案：
			1、插入到叶子节点之下，不需要动其他节点，速度快，但形成的不是完全二叉查找树。
			2、严格遵守完全二叉查找树，需要移动其他节点，速度慢。
		由于题目并不要求完全二叉查找树，所以我们取方案一即可。
	时间复杂度：O(n)
		最坏情况下，我们需要将值插入到树的最深的叶子结点上，而叶子节点最深为 O(n)。
	空间复杂度：O(logn)
*/
func insertIntoBST(root *TreeNode, val int) *TreeNode {
	if root == nil {
		return &TreeNode{Val : val}
	}
	insert(root, val)
	return root
}
func insert(root *TreeNode, val int) {
	if val < root.Val && root.Left == nil {
		root.Left = &TreeNode{Val : val}
	}
	if val > root.Val && root.Right == nil {
		root.Right = &TreeNode{Val : val}
	}
	if val < root.Val {
		insert(root.Left, val)
	} else {
		insert(root.Right, val)
	}
}

/* 
	方法二：DFS-自底向上
	思路：
		严格遵守完全二叉查找树，重构每一颗子树的左右节点。
	时间复杂度：O(n)
		最坏情况下，我们需要将值插入到树的最深的叶子结点上，而叶子节点最深为 O(n)。
	空间复杂度：O(logn)
*/
func insertIntoBST2(root *TreeNode, val int) *TreeNode {
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

/* 
===================== 3、删除二叉搜索树中的节点 =====================
给定一个二叉搜索树的根节点 root 和一个值 key，删除二叉搜索树中的 key 对应的节点，
并保证二叉搜索树的性质不变。返回二叉搜索树（有可能被更新）的根节点的引用。

一般来说，删除节点可分为两个步骤：
    首先找到需要删除的节点；
    如果找到了，删除它。

说明： 要求算法时间复杂度为 O(h)，h 为树的高度。

示例:
root = [5,3,6,2,4,null,7]
key = 3
    5
   / \
  3   6
 / \   \
2   4   7

给定需要删除的节点值是 3，所以我们首先找到 3 这个节点，然后删除它。
一个正确的答案是 [5,4,6,2,null,null,7], 如下图所示。
    5
   / \
  4   6
 /     \
2       7

另一个正确答案是 [5,2,6,null,4,null,7]。
    5
   / \
  2   6
   \   \
    4   7

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/delete-node-in-a-bst
*/
/* 
	算法：DFS
	思路：
		删除二叉搜索树的节点有三种情况：
			1、被删除节点只有左子树，则将其返回给被删除节点的父节点
			2、被删除节点只有右子树，则将其返回给被删除节点的父节点
			3、若被删除节点同时有左右子树，则在右子树中查找最左节点，再把左子树挂上去
	时间复杂度：O(H)
		H 是树的高度。在算法的执行过程中，我们一直在树上向左或向右移动。
		首先先用 O(H1) 的时间找到要删除的节点，H1​ 值得是从根节点到要删除节点的高度。
		然后删除节点需要 O(H2) 的时间，H2​ 指的是从要删除节点到替换节点的高度。
		由于 O(H1+H2)=O(H)，H 值得是树的高度，若树是一个平衡树则 H = log⁡N。
	空间复杂度：O(H)
		递归时堆栈使用的空间，H 是树的高度。
*/
func deleteNode(root *TreeNode, key int) *TreeNode {
	if root == nil {
		return nil
	}
	if key < root.Val {
		root.Left = deleteNode(root.Left, key)
	} else if key > root.Val {
		root.Right = deleteNode(root.Right, key)
	} else {
		// root.Val == key，找到要删除的节点
		if root.Left == nil {
			return root.Right
		} 
		if root.Right == nil {
			return root.Left
		}
		// 同时存在左右子树，从右子树中找最左边的节点，然后把左子树挂上去
		curNode := root.Right
		for curNode.Left != nil {
			curNode = curNode.Left
		}
		// 挂载左子树
		curNode.Left = root.Left
		// 返回挂载后的右子树（把左子树挂到了右子树上，返回右子树，即删除了当前节点）
		return root.Right
	}
	return root
}

/* 
===================== 4、平衡二叉树 =====================
给定一个二叉树，判断它是否是高度平衡的二叉树。
本题中，一棵高度平衡二叉树定义为：
    一个二叉树每个节点 的左右两个子树的高度差的绝对值不超过 1 。

示例 1：
输入：root = [3,9,20,null,null,15,7]
输出：true

示例 2：
输入：root = [1,2,2,3,3,null,null,4,4]
输出：false

示例 3：
输入：root = []
输出：true

提示：
    树中的节点数在范围 [0, 5000] 内
    -104 <= Node.val <= 104

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/balanced-binary-tree
*/
/* 
	方法一：DFS
	思路：
		利用自定向上DFS求树的高度，在求高度的过程中判断每一颗子树是否平衡。
	时间复杂度：O(H)
		H 是树的高度
	空间复杂度：O(H)
*/
func isBalanced(root *TreeNode) bool {
	if root == nil {
		return true
	}
	var depathAndBalance func(*TreeNode) (int, bool)
	depathAndBalance = func(root *TreeNode) (int, bool) {
		if root == nil {
			return 0, true
		}
		li, lb := depathAndBalance(root.Left)
		ri, rb := depathAndBalance(root.Right)
		if lb && rb && abs(li, ri) < 2 {
			return max(li, ri) + 1, true
		}
		return max(li, ri) + 1, false
	}
	_, res := depathAndBalance(root)
	return res
}
func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
func abs(a, b int) int {
	if a > b {
		return a - b
	}
	return b - a
}

// ===================== 案例测试 =====================
/* 
	 2
	/ \
   1   4
  	  / \
 	 3   5
*/
func buildTree() *TreeNode {
	node1 := &TreeNode{Val : 1}
	node2 := &TreeNode{Val : 2}
	node3 := &TreeNode{Val : 3}
	node4 := &TreeNode{Val : 4}
	node5 := &TreeNode{Val : 5}
	
	node2.Left = node1
	node2.Right = node4
	node4.Left = node3
	node4.Right = node5
	return node2
}


// 1、验证二叉搜索树
func isValidBSTTest() {
	root := buildTree()
	res := isValidBST2(root)
	fmt.Println(res)
}

// 2、二叉搜索树中的插入操作
func insertIntoBSTTest() {
	root := buildTree()
	res := insertIntoBST2(root, 6)
	inOrderPrint(res)
}
func inOrderPrint(root *TreeNode) {
	if root == nil {
		return
	}
	inOrderPrint(root.Left)
	fmt.Printf("%v ", root.Val)
	inOrderPrint(root.Right)
}

// 3、删除二叉搜索树中的节点
func deleteNodeTest() {
	root := buildTree()
	res := deleteNode(root, 2)
	inOrderPrint(res)
}

// 4、平衡二叉树
func isBalancedTest() {
	root := buildTree()
	res := isBalanced(root)
	fmt.Println(res)
}

func main() {
	// isValidBSTTest()
	// insertIntoBSTTest()
	// deleteNodeTest()
	isBalancedTest()
}