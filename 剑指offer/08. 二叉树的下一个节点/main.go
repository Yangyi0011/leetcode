package main

import (
	"fmt"
)

/*
============== 剑指 Offer 08. 二叉树的下一个节点 ==============
给定一颗二叉树和其中的一个节点，如何找出中序遍历序列的下一个节点？
树中的节点除了有两个分别指向左右子节点的指针外，还有一个指向父节点
的指针。

示列：
			5
		  /    \
		 3		6
	   /  \      \
	  2	   4	  7
	/
   1

输入：
[1, 2, 3, 4, 5, 6, 7]
4
输出：
5
说明：
节点 4 在中序遍历序列中的下一个节点是 5

注：在输入的二叉树中没有重复节点。
*/
type TreeNode struct {
	Val                 int
	Parent, Left, Right *TreeNode
}
/* 
	方法一：遍历
	思路：
		寻找某一个节点中序遍历序列中的下一个节点有以下三种情况：
			1、该节点有右节点，则右子树的最左节点是它的下一个节点。
			2、没有右节点，但该节点是它的父节点的左节点，则它的父节点是它的
				下一个节点。
			3、既没有右节点，也不是它的父节点的左节点，则需要通过 Parent 指
				针向上遍历，直到找到一个节点是它的父节点的左节点为止，如果这
				么一个节点存在，那么它的父节点即为我们要找的下一个节点。
		时间复杂度：
		空间复杂度：O(1)
*/
func findNext(root *TreeNode, target *TreeNode) *TreeNode {
	if root == nil || target == nil { 
		return nil
	}
	// 有右节点，则右子树的最左节点是它的下一个节点
	if target.Right != nil {
		next := target.Right
		for next.Left != nil {
			next = next.Left
		}
		return next
	}
	// 是父节点的左节点，则父节点是它的下一个节点
	if target.Parent != nil && target.Parent.Left == target {
		return target.Parent
	}
	// 即没有右节点，且不是父节点的左节点，则需要向上遍历找到一个是父节点的
	// 左节点的节点，如果此节点存在，那么此节点的父节点是我们要找的下一个节点
	cur := target
	for cur.Parent != nil {
		if cur.Parent.Left == cur {
			return cur.Parent
		}
		cur = cur.Parent
	}
	return nil
}

func main() {
	n1 := &TreeNode{Val : 1}
	n2 := &TreeNode{Val : 2}
	n3 := &TreeNode{Val : 3}
	n4 := &TreeNode{Val : 4}
	n5 := &TreeNode{Val : 5}
	n6 := &TreeNode{Val : 6}
	n7 := &TreeNode{Val : 7}

	n5.Left = n3
	n5.Right = n6
	n3.Left = n2
	n3.Right = n4
	n3.Parent = n5
	n2.Left = n1
	n2.Parent = n3
	n1.Parent = n2
	n4.Parent = n3
	n6.Right = n7
	n6.Parent = n5
	n7.Parent = n6

	root := n5
	res := findNext(root, n4)
	fmt.Println(res)
}