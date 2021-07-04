package main

import "fmt"

/*
============== 剑指 Offer 22. 链表中倒数第k个节点 ==============
输入一个链表，输出该链表中倒数第k个节点。为了符合大多数人的习惯，本题从1开始计数，
即链表的尾节点是倒数第1个节点。

例如，一个链表有 6 个节点，从头节点开始，它们的值依次是 1、2、3、4、5、6。这个链
表的倒数第 3 个节点是值为 4 的节点。

示例：
给定一个链表: 1->2->3->4->5, 和 k = 2.
返回链表 4->5.

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/lian-biao-zhong-dao-shu-di-kge-jie-dian-lcof
*/
// Definition for singly-linked list.
type ListNode struct {
	Val  int
	Next *ListNode
}
/* 
	方法一：双指针
	思路：
		建立 h1、h2 两个指针，h2 先走 k 步，之后 h1、h2 再一起走，当 h2 指向 nil
		时，h1 指向的位置即为倒数第 k 个节点。
	时间复杂度：O(n)
		n 为链表节点个数，两个指针最多遍历一次链表
	空间复杂度：O(1)
*/
func getKthFromEnd(head *ListNode, k int) *ListNode {
	if head == nil {
		return nil
	}
	h1, h2 := head, head
	// h2 先走 k 步
	for k > 0 && h2 != nil {
		h2 = h2.Next
		k--
	}
	// 如果 k 还大于 0，说明链表长度小于 k，返回 nil
	if k > 0 {
		return nil
	}
	// h1、h2 一起走，当 h2 == nil 时，h1 指向的即为倒数第 k 个节点
	for h2 != nil {
		h1 = h1.Next
		h2 = h2.Next
	}
	return h1
}

func main() {
	n1 := &ListNode{Val: 1}
	n2 := &ListNode{Val: 2}
	n3 := &ListNode{Val: 3}
	n4 := &ListNode{Val: 4}
	n5 := &ListNode{Val: 5}

	n1.Next = n2
	n2.Next = n3
	n3.Next = n4
	n4.Next = n5

	res := getKthFromEnd(n1, 3)
	for res != nil {
		fmt.Printf("%v->", res.Val)
		res = res.Next
	}
	fmt.Println("nil")
}
