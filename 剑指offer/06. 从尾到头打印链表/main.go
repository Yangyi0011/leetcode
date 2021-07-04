package main

/*
============== 剑指 Offer 06. 从尾到头打印链表 ==============
输入一个链表的头节点，从尾到头反过来返回每个节点的值（用数组返回）。

示例 1：
输入：head = [1,3,2]
输出：[2,3,1]

限制：
0 <= 链表长度 <= 10000

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/cong-wei-dao-tou-da-yin-lian-biao-lcof
*/
type ListNode struct {
	Val  int
	Next *ListNode
}

/*
	方法一：反转数组
	思路：
		从头到尾遍历链表，把每一个值放在数组中，遍历完成时把数组反转进行返回。
	时间复杂度：O(n)
		n 是链表节点个数，我们遍历链表需要 O(n) 的时间，反转数组需要 O(n/2) 
		的时间，所以总的线性时间复杂度为 O(n)。
	空间复杂度：O(n)
		n 是链表节点个数，我们需要用长度为 n 的数组来存储遍历结果并返回。
*/
func reversePrint(head *ListNode) []int {
	if head == nil {
		return []int{}
	}
	ans := []int{}
	for head != nil {
		ans = append(ans, head.Val)
		head = head.Next
	}
	n := len(ans)
	for i := 0; i < (n >> 1); i ++ {
		ans[i], ans[n - 1 - i] = ans[n - 1 - i], ans[i]
	}
	return ans
}

/* 
	方法二：反转链表
	思路：
		如果原链表可以修改，那么我们可以先反转原链表再进行遍历。
	时间复杂度：O(n)
		n 是链表节点个数，反转链表需要 O(n) 的时间，遍历反转后的链表也
		需要 O(n) 的时间，所以总的线性时间复杂度为 O(n)。
	空间复杂度：O(n)
		n 是链表节点个数，我们需要用长度为 n 的数组来存储遍历结果并返回。
*/
func reversePrint2(head *ListNode) []int {
	if head == nil {
		return []int{}
	}
	var pre *ListNode = nil
	for head != nil {
		next := head.Next
		head.Next = pre
		pre = head
		head = next
	}
	ans := []int{}
	for pre != nil {
		ans = append(ans, pre.Val)
		pre = pre.Next
	}
	return ans
}

/* 
	方法三：使用栈
	思路：
		前面两种方法中，方法一属于投机取巧，有点不符合题意，而方法二改变了
		原数组，如果在面试中被面试官严格限制，则两种方法都不能用。
		因为链表只能通过 Next 指针向后而不能先前，但题目又要求我们从尾到头
		打印链表，即先进后出的顺序，如此我们可以想到要用栈。
	时间复杂度：O(n)
		n 是链表节点个数，我们遍历链表需要 O(n) 的时间，从栈中取出结果也需
		要 O(n) 的时间，所以总的线性时间复杂度为 O(n)。
	空间复杂度：O(n)
		n 是链表节点个数，我们需要用栈来存储从头到尾遍历的链表元素，需要 O(n)
		的额外空间，我们需要用数组来存储 n 个节点的值进行返回，需要 O(n) 的
		额外空间，所以总的线性空间复杂度是 O(n)。
*/
func reversePrint3(head *ListNode) []int {
	if head == nil {
		return []int{}
	}
	cur := head
	stack := make([]*ListNode, 0)
	for cur != nil {
		stack = append(stack, cur)
		cur = cur.Next
	}
	ans := make([]int, len(stack))
	for i := 0; i < len(ans); i ++ {
		ans[i] = stack[len(stack) - 1].Val
		stack = stack[: len(stack) - 1]
	}
	return ans
}