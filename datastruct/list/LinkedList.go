package list

import (
	"strconv"
)

/* 
	设计单链表，实现这些功能：
		get(index)：获取链表中第 index 个节点的值。如果索引无效，则返回-1。
		addAtHead(val)：在链表的第一个元素之前添加一个值为 val 的节点。插入后，新节点将成为链表的第一个节点。
		addAtTail(val)：将值为 val 的节点追加到链表的最后一个元素。
		addAtIndex(index,val)：在链表中的第 index 个节点之前添加值为 val  的节点。
			如果 index 等于链表的长度，则该节点将附加到链表的末尾。
			如果 index 大于链表长度，则不会插入节点。如果index小于0，则在头部插入节点。
		deleteAtIndex(index)：如果索引 index 有效，则删除链表中的第 index 个节点。

		注：index 从 0 开始

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/linked-list/jy291/
*/
type Node struct {
	Val int
	Next *Node
}

type LinkedList struct {
	Head *Node
	Size int
}

// get(index)：获取链表中第 index 个节点的值。如果索引无效，则返回-1。
func (this *LinkedList) Get(index int) int {
	if index >= this.Size || index < 0 {
		return -1
	}
	p := this.Head
	// 找到第 index 个节点的上一个节点
	for i := 0; i < index; i ++ {
		p = p.Next
	}
	return p.Next.Val
}

// addAtHead(val)：在链表的第一个元素之前添加一个值为 val 的节点。
// 插入后，新节点将成为链表的第一个节点。
func (this *LinkedList) AddAtHead(v int) {
	this.AddAtIndex(0, v)
}

// addAtTail(val)：将值为 val 的节点追加到链表的最后一个元素。
func (this *LinkedList) AddAtTail(v int) {
	this.AddAtIndex(this.Size, v)
}

// addAtIndex(index,val)：在链表中的第 index 个节点之前添加值为 val  的节点。
// 如果 index 等于链表的长度，则该节点将附加到链表的末尾。如果 index 大于链表长度，
// 则不会插入节点。如果index小于0，则在头部插入节点。
func (this *LinkedList) AddAtIndex(index, v int) {
	if index > this.Size {
		return
	}
	node := &Node{Val:v}
	p := this.Head
	for i := 0; i < index; i ++ {
		p = p.Next
	}
	node.Next = p.Next
	p.Next = node
	this.Size ++
}

// deleteAtIndex(index)：如果索引 index 有效，则删除链表中的第 index 个节点。
func (this *LinkedList) DeleteAtIndex(index int) {
	if index >= this.Size || index < 0 {
		return
	}
	p := this.Head
	// 找到 index 的上一个节点
	for i := 0; i < index; i ++ {
		p = p.Next
	}
	// 删除 index 节点
	p.Next = p.Next.Next
	this.Size --
}

func (this LinkedList) String() string {
	p := this.Head
	str := ""
	for p.Next != nil {
		str += strconv.Itoa(p.Next.Val)
		str += "->"
		p = p.Next
	}
	return str
}

func NewLinkedList() *LinkedList {
	return &LinkedList{Head : &Node{}, Size : 0}
}
