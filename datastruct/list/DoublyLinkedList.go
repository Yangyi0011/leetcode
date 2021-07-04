package list

/* 
	双向链表
*/

/* 
	设计双向链表，实现这些功能：
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
type DoublyListNode struct {
	Val int
	Prev, Next *DoublyListNode
}

type DoublyLinkedList struct {
	Head *DoublyListNode
	Size int
}

// get(index)：获取链表中第 index 个节点的值。如果索引无效，则返回-1。
func (this *DoublyLinkedList) Get(index int) int {
	if index >= this.Size || index < 0 {
		return -1
	}
	p := this.Head
	// 因为 head 是前置虚拟节点，所以找到的是 index 的上一个节点
	for i := 0; i < index; i ++ {
		p = p.Next
	}
	return p.Next.Val
}

// addAtHead(val)：在链表的第一个元素之前添加一个值为 val 的节点。
// 插入后，新节点将成为链表的第一个节点。
func (this *DoublyLinkedList) AddAtHead(val int) {
	this.AddAtIndex(0, val)
}

// addAtTail(val)：将值为 val 的节点追加到链表的最后一个元素。
func (this *DoublyLinkedList) AddAtTail(val int) {
	this.AddAtIndex(this.Size, val)
}

// addAtIndex(index,val)：在链表中的第 index 个节点之前添加值为 val  的节点。
// 如果 index 等于链表的长度，则该节点将附加到链表的末尾。如果 index 大于链表长度，
// 则不会插入节点。如果index小于0，则在头部插入节点。
func (this *DoublyLinkedList) AddAtIndex(index int, val int) {
	if index > this.Size {
		return
	}
	p := this.Head
	// 因为 head 是前置虚拟节点，所以找到的是 index 的上一个节点
	for i := 0; i < index; i ++ {
		p = p.Next
	}
	node := &DoublyListNode{Val:val}
	node.Next = p.Next
	if p.Next != nil {
		p.Next.Prev = node
	}
	node.Prev = p
	p.Next = node
	this.Size ++
}

// deleteAtIndex(index)：如果索引 index 有效，则删除链表中的第 index 个节点。
func (this *DoublyLinkedList) DeleteAtIndex(index int) {
	if index < 0 || index >= this.Size {
		return
	}
	p := this.Head
	// 因为 head 是前置虚拟节点，所以找到的是 index 的上一个节点
	for i := 0; i < index; i ++ {
		p = p.Next
	}
	// 因为找到的是 index 的上一个，所以 p.Next 必然是存在的
	if p.Next.Next != nil {
		p.Next.Next.Prev = p
	}
	p.Next = p.Next.Next
	this.Size --
}

type MyLinkedList struct {
	DoublyLinkedList
}

func Constructor() *MyLinkedList {
	return &MyLinkedList{
		DoublyLinkedList : DoublyLinkedList{Head : &DoublyListNode{}, Size : 0},
	}
}
