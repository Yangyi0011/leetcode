package list

/* 
	反转链表：

*/

/* 
========================== 1、反转链表 ==========================
反转一个单链表。

示例:
输入: 1->2->3->4->5->NULL
输出: 5->4->3->2->1->NULL

进阶:
你可以迭代或递归地反转链表。你能否用两种方法解决这道题？

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/reverse-linked-list
*/
/* 
	方法一：迭代
	思路：
		使用一个 空 节点当做链表的上一个节点，迭代时需要记录好下一个节点，
		然后把当前节点的 next 指针指向上一个节点就行了
	时间复杂度：O(n)
		n 表示链表节点的个数
	空间复杂度：O(1)
*/
func reverseList1(head *ListNode) *ListNode {
	if head == nil || head.Next == nil {
		return head
	}
	var pre *ListNode
	for head != nil {
		// 记录下一个节点及其剩余节点
		next := head.Next
		// 当前节点的 next 指针进行反转
		head.Next = pre
		// 向后传递
		pre = head
		head = next
	}
	return pre
}

/* 
	方法二：递归
	思路：
		深度优先，自底向上进行链表反转，即先反转当前节点的下一个节点，
		如果下一个节点为空，返回当前节点.
		难点：		   
			1->2->3-> 4<-5
					  ↓
					  nil
		假如4、5节点已经反转完成，现在要反转当前节点 3，在递归中该怎么做呢？
		已知当前节点 3 的 next 指针还是指向 4，而 4 已经反转完成，返回 5->4->nil
		此时只需做：cur.Next.Next = cur，就完成了节点 3 的反转，从而返回
		5->4->3->nil 去处理节点 2 了。
	时间复杂度：O(n)
		n 表示链表节点的个数，我们需要递归处理每一个节点
	空间复杂度：O(n)
		n 表示链表节点的个数，递归过程中需要用到函数栈
*/
func reverseList2(head *ListNode) *ListNode {
	if head == nil || head.Next == nil {
		return head
	}
	// 优先处理下一个节点
	res := reverseList2(head.Next)
	// 处理当前节点
	head.Next.Next = head
	// 因为当前节点的下一个节点已经处理完成，所以需要指向 nil，避免出现环
	head.Next = nil
	return res
}

/* 
========================== 2、移除链表元素 ==========================
删除链表中等于给定值 val 的所有节点。

示例:
输入: 1->2->6->3->4->5->6, val = 6
输出: 1->2->3->4->5

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/linked-list/f9izv/
*/
/* 
	方法一：迭代
	思路：
		借助于一个哑结点当做起始节点，由此判断下一个节点是否要删除。
	时间复杂度：O(n)
		n 表示链表节点的个数，我们需要迭代处理每一个节点
	空间复杂度：O(1)
*/
func removeElements1(head *ListNode, val int) *ListNode {
	if head == nil {
		return nil
	}
	pre := &ListNode{}
	pre.Next = head
	p := pre
	for p != nil && p.Next != nil {
		// 删除节点
		if p.Next.Val == val {
			p.Next = p.Next.Next
		} else {
			// 必须用 else 向后处理，否则无法删除多个连续节点
			// 即删除完成后不向下走，继续处理当前节点的下一个元素，
			// 因为删除后下一个元素已经改变了
			p = p.Next
		}
	}
	// 不能直接返回 head，因为 head 也有可能被删除
	return pre.Next
}

/* 
	方法二：递归
	思路：
		深度优先，优先处理下一个节点，当当前节点要被删除时，
		返回当前节点的下一个节点，当当前节点为 nil 时，返回 nil
	时间复杂度：O(n)
	空间复杂度：O(n)
*/
func removeElements2(head *ListNode, val int) *ListNode {
	if head == nil {
		return nil
	}
	head.Next = removeElements2(head.Next, val)
	if head.Val == val {
		return head.Next
	} else {
		return head
	}
}

/* 
========================== 3、奇偶链表 ==========================
给定一个单链表，把所有的奇数节点和偶数节点分别排在一起。请注意，这里的奇数
节点和偶数节点指的是节点编号的奇偶性，而不是节点的值的奇偶性。
请尝试使用原地算法完成。你的算法的空间复杂度应为 O(1)，时间复杂度应
为 O(nodes)，nodes 为节点总数。

示例 1:
输入: 1->2->3->4->5->NULL
输出: 1->3->5->2->4->NULL

示例 2:
输入: 2->1->3->5->6->4->7->NULL 
输出: 2->3->6->7->1->5->4->NULL

说明:
    应当保持奇数节点和偶数节点的相对顺序。
    链表的第一个节点视为奇数节点，第二个节点视为偶数节点，以此类推。

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/odd-even-linked-list
*/
/* 
	方法一：分离节点后合并
	思路：
		分离奇数位节点和偶数位节点成奇链表和偶链表，
		最后再奇链表的尾部去连接偶链表的头部。
	时间复杂度：O(n)
		n 是原链表的节点个数
	空间复杂度：O(1)
		我们只借助于常数级的变量空间
*/
func oddEvenList(head *ListNode) *ListNode {
	if head == nil || head.Next == nil {
		return head
	}
	// 奇偶链表的头部
	addHead := head
	evenHead := head.Next
	// 指向奇偶链表的指针
	add := addHead
	even := evenHead
	// 因为每次处理都是先处理奇位节点，偶位节点后处理，且偶位节点之后还
	// 连接着剩余节点，所以要用偶位节点来做遍历判断
	for even != nil && even.Next != nil {
		// 下一个奇节点在当前偶节点的后面
		add.Next = even.Next
		// 奇指针后移
		add = add.Next
		// 下一个偶节点在处理后的奇节点的后面
		even.Next = add.Next
		// 偶指针后移
		even = even.Next
	}
	// 奇连接的尾部连接偶链表的头部
	add.Next = evenHead
	// 返回奇链表的头部
	return addHead
}

