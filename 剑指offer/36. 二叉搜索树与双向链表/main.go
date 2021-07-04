package main

import (
	"fmt"
)

/*
============== 剑指 Offer 22. 链表中倒数第k个节点 ==============
输入一棵二叉搜索树，将该二叉搜索树转换成一个排序的循环双向链表。要求不能创建
任何新的节点，只能调整树中节点指针的指向。

为了让您更好地理解问题，以下面的二叉搜索树为例：
		4
	   / \
	  2   5
	 / \
	1	3

我们希望将这个二叉搜索树转化为双向循环链表。链表中的每个节点都有一个前驱和
后继指针。对于双向循环链表，第一个节点的前驱是最后一个节点，最后一个节点的后
继是第一个节点。

下图展示了上面的二叉搜索树转化成的链表。“head” 表示指向链表中有最小元素的节点。

  head
    ↓
	1 <——> 2 <——> 3 <——> 4 <——> 5
	↑|							↑|
	|↓——————————————————————————||
	|————————————————————————————↓

特别地，我们希望可以就地完成转换操作。当转化完成以后，树中节点的左指针需要指向
前驱，树中节点的右指针需要指向后继。还需要返回链表中的第一个节点的指针。

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/er-cha-sou-suo-shu-yu-shuang-xiang-lian-biao-lcof
*/
type Node struct {
	Val         int
	Left, Right *Node
}

/*
	方法一：递归中序遍历
	思路：
		根据二叉搜索树的性质，我们只需要对二叉搜索树进行中序遍历就可以得到一个
		有序链表，而要形成双向链表，我们需要在遍历过程中对指针进行一些操作。
		对于每一个节点：
			1、优先处理其左子树
			2、记录当前节点的前驱节点。在中序遍历过程中，上一个被处理的节点
				即为当前节点的前驱节点。
			3、把当前节点的 Left 指针指向前驱节点，把前驱节点的 Right 指针
				指向当前节点，完成双向链表的连接操作。
			4、继续处理右子树。
			5、最后在返回之前，遍历链表，找到链表的头、尾结点，把链表的头结
				点和尾结点连接，使之形成循环双向链表。
	时间复杂度：O(n)
		n 是二叉搜索树的节点个数，我们需要遍历树的每一个节点进行处理。
	空间复杂度：O(n)
		n 是二叉搜索树的节点个数，递归需要消耗与树的高度相同的额外栈空间，最
		坏情况下二叉树的所有节点连城一个链表，此时的空间复杂度为 O(n)。
*/
func treeToDoublyList(root *Node) *Node {
	if root == nil {
		return root
	}
	// 记录当前节点的前驱节点
	var pre *Node
	var DFS func(cur *Node)
	DFS = func(cur *Node) {
		if cur == nil {
			return
		}
		// 处理当前节点的左子树
		DFS(cur.Left)
		// 将当前节点的前驱节点与当前节点相连
		if pre != nil {
			pre.Right = cur
			cur.Left = pre
		}
		// 记录当前节点的前驱节点
		pre = cur
		// 处理右子树
		DFS(cur.Right)
	}
	DFS(root)
	// 找到双向链表的头、尾结点
	head := root
	for head.Left != nil {
		head = head.Left
	}
	tail := root
	for tail.Right != nil {
		tail = tail.Right
	}
	// 连接头结点和尾结点使双向链表循环
	head.Left = tail
	tail.Right = head
	return head
}

/* 
	方法二：循环中序遍历
	思路：
		同上。
	时间复杂度：O(n)
		n 是二叉搜索树的节点个数，我们需要遍历树的每一个节点进行处理。
	空间复杂度：O(n)
		n 是二叉搜索树的节点个数，我们需要的栈最多需要存储 n 个节点。
*/
func treeToDoublyList2(root *Node) *Node {
	if root == nil {
		return root
	}
	// 记录当前节点的前驱节点
	var pre *Node
	stack := make([]*Node, 0);
	cur := root
	for cur != nil || len(stack) > 0 {
		// 优先处理左子树
		for cur != nil {
			stack = append(stack, cur)
			cur = cur.Left
		}
		cur = stack[len(stack) - 1]
		stack = stack[: len(stack) - 1]
		// 将当前节点的前驱节点与当前节点相连
		if pre != nil {
			pre.Right = cur
			cur.Left = pre
		}
		// 记录前驱节点
		pre = cur
		// 处理右子树
		cur = cur.Right
	}
	// 找到双向链表的头、尾结点
	head := root
	for head.Left != nil {
		head = head.Left
	}
	tail := root
	for tail.Right != nil {
		tail = tail.Right
	}
	// 连接头结点和尾结点使双向链表循环
	head.Left = tail
	tail.Right = head
	return head
}

/* 
	方法三：循环优化
	思路：
		在上面两个方法中，我们需要在中序遍历处理完成后再去寻找链表的头节点
		和尾节点，此时需要花费 O(n) 的时间，对此，我们可以把对头结点和尾节点
		的寻找放到中序遍历过程中去处理，以提高效率。
	时间复杂度：O(n)
	空间复杂度：O(n)
*/
func treeToDoublyList3(root *Node) *Node {
	if root == nil {
		return root
	}
	// 记录当前节点的前驱节点以及双向链表的头节点
	var pre, head *Node
	stack := make([]*Node, 0);
	cur := root
	for cur != nil || len(stack) > 0 {
		// 优先处理左子树
		for cur != nil {
			stack = append(stack, cur)
			cur = cur.Left
		}
		cur = stack[len(stack) - 1]
		stack = stack[: len(stack) - 1]
		// 将当前节点的前驱节点与当前节点相连
		if pre != nil {
			pre.Right = cur
			cur.Left = pre
		} else {
			// 如果 pre 为 nil，说明当前节点是双向链表的头结点，只会赋值一次
			head = cur
		}
		// 记录前驱节点
		pre = cur
		// 处理右子树
		cur = cur.Right
	}
	// 中序遍历处理完时，pre 记录的是最后一个被处理的节点，即为双向链表
	// 的尾结点
	// 连接头结点和尾结点使双向链表循环
	head.Left = pre
	pre.Right = head
	return head
}

func main() {
	n1 := &Node{Val: 1}
	n2 := &Node{Val: 2}
	n3 := &Node{Val: 3}
	n4 := &Node{Val: 4}
	n5 := &Node{Val: 5}

	n4.Left = n2
	n4.Right = n5
	n2.Left = n1
	n2.Right = n3

	res := treeToDoublyList3(n4)
	node := res
	for node != nil {
		fmt.Printf("%v->", node.Val)
		node = node.Right
	}
	fmt.Println()
}
