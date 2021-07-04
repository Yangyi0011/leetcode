package main

/* 
============== 剑指 Offer 07. 重建二叉树 ==============
输入某二叉树的前序遍历和中序遍历的结果，请重建该二叉树。假设输入的前序遍
历和中序遍历的结果中都不含重复的数字。

例如，给出

前序遍历 preorder = [3,9,20,15,7]
中序遍历 inorder = [9,3,15,20,7]

返回如下的二叉树：

    3
   / \
  9  20
    /  \
   15   7

限制：
0 <= 节点个数 <= 5000

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/zhong-jian-er-cha-shu-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
*/
type TreeNode struct {
	Val int
	Left *TreeNode
	Right *TreeNode
}
/* 
	方法一：递归
	思路：
		由前序遍历的性质可知，前序遍历数组中的第一个元素是树的根节点，而在中序遍历
		数组中，根节点前面的元素是根节点的左子树，根节点后面的元素是根节点的右子树，
		可以由此性质来递归构建二叉树。
		为了提高在中序遍历数组中寻找根节点下标的效率，我们需要借助哈希表来预先
		存储中序遍历数组。
		在构建根节点子树时，需要先构建左子树，再构建右子树：
			在前序遍历的数组中整个数组是先存储根节点，再存储左子树的节点，最后存储
			右子树的节点，如果按每次选择「前序遍历的第一个节点」为根节点，则先被构
			造出来的应该为左子树。
	时间复杂度：O(n)
		其中 n 是树中的节点个数。
	空间复杂度：O(n)
		我们需要使用 O(n) 的空间存储哈希表，以及 O(h)（其中 h 是树的高度）的空间表示
		递归时栈空间。这里 h<n，所以总空间复杂度为 O(n)。
*/
func buildTree2(preorder []int, inorder []int) *TreeNode {
	if len(preorder) == 0 || len(inorder) == 0 {
		return nil
	}
    // 中序遍历数组的 value-index map
	idxMap := make(map[int]int, len(inorder))
	for i, v := range inorder {
		idxMap[v] = i
	}
	// 在闭包函数内部引用外部数据时，如果该数据不在闭包函数的形参列表内，则
	// 传递到闭包内部的是外部数据的引用【引用传递】
	var build func(inorderLeftIdx, inorderRightIdx int) *TreeNode
	build = func(inorderLeftIdx, inorderRightIdx int) *TreeNode {
		// 已经没有剩余节点需要处理
		if inorderLeftIdx > inorderRightIdx {
			return nil
		}
		// 从前序遍历序列中取第一个节点作为当前子树的根节点
		val := preorder[0]
		preorder = preorder[1:]
		// 获取该根节点的值在中序遍历中的下标
		idx := idxMap[val]
		// 构建该根节点和它的左右子树
		root := &TreeNode{Val : val}
		// 在中序遍历序列中，根节点下标的左边是它的左子树节点
		root.Left = build(inorderLeftIdx, idx - 1)
		// 在中序遍历序列中，根节点下标的右边是它的右子树节点
		root.Right = build(idx + 1, inorderRightIdx)
		return root
	}
	return build(0, len(preorder) - 1)
}

/* 
	方法二：迭代
	思路：
		对于前序遍历中的任意两个连续节点 u 和 v，根据前序遍历的流程，我们可
		以知道 u 和 v 只有两种可能的关系：
			v 是 u 的左儿子。这是因为在遍历到 u 之后，下一个遍历的节点就
			是 u 的左儿子，即 v；
			u 没有左儿子，并且 v 是 u 的某个祖先节点（或者 u 本身）的右儿子。
			如果 u 没有左儿子，那么下一个遍历的节点就是 u 的右儿子。如果 u 
			没有右儿子，我们就会向上回溯，直到遇到第一个有右儿子
			（且 u 不在它的右儿子的子树中）的节点 ua​，那么 v 就是 ua​ 的右儿子。
	时间复杂度：O(n)
		其中 n 是树中的节点个数。
	空间复杂度：O(n)
		我们需要使用 O(n) 的空间存储哈希表，以及 O(h)（其中 h 是树的高度）的空间表示
		递归时栈空间。这里 h<n，所以总空间复杂度为 O(n)。
*/
func buildTree(preorder []int, inorder []int) *TreeNode {
    if len(preorder) == 0 {
        return nil
    }
	// 先序遍历的第一个节点是根节点
    root := &TreeNode{preorder[0], nil, nil}
    stack := []*TreeNode{}
    stack = append(stack, root)
	// 记录中序下标，从 0 开始
    var inorderIndex int
    for i := 1; i < len(preorder); i++ {
        preorderVal := preorder[i]
        node := stack[len(stack)-1]
		// 如果栈顶元素与当前中序元素不相等，则把当前遍历的先序元素当做栈顶
		// 元素的左子树节点入栈
        if node.Val != inorder[inorderIndex] {
            node.Left = &TreeNode{preorderVal, nil, nil}
            stack = append(stack, node.Left)
        } else {
			// 否则弹出所有与当前中序元素相等的栈顶元素，把当前遍历的先序元素
			// 当做栈顶元素的右子树入栈
            for len(stack) != 0 && stack[len(stack)-1].Val == inorder[inorderIndex] {
                node = stack[len(stack)-1]
                stack = stack[:len(stack)-1]
                inorderIndex++
            }
            node.Right = &TreeNode{preorderVal, nil, nil}
            stack = append(stack, node.Right)
        }
    }
    return root
}
