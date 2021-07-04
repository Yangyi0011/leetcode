package list

import (
	"fmt"
)

/* 
	双指针技巧
	快慢双指针模板：
		fast, slow := head, head
		for slow != nil && fast != nil && fast.Next != nil {
			slow = slow.Next
			fast = fast.Next.Next
			// 相遇
			if slow = fast {
				return true
			}
		}
		return false

	注意点：
		1. 在调用 next 字段之前，始终检查节点是否为空。
			获取空节点的下一个节点将导致空指针错误。例如，
			在我们运行 fast = fast.next.next 之前，需要检查 
			fast 和 fast.next 不为空。
		2. 仔细定义循环的结束条件，别死循环了。
	复杂度分析：
		空间复杂度分析容易。如果只使用指针，而不使用任何其他额外的空间，
		那么空间复杂度将是 O(1)。但是，时间复杂度的分析比较困难。
		为了得到答案，我们需要分析运行循环的次数。
		在前面的查找循环示例中，假设我们每次移动较快的指针 2 步，每次移动较慢的指针 1 步。
			如果没有循环，快指针需要 N/2 次才能到达链表的末尾，其中 N 是链表的长度。
			如果存在循环，则快指针需要 M 次才能赶上慢指针，其中 M 是列表中循环的长度。
		显然，M <= N 。所以我们将循环运行 N 次。对于每次循环，我们只需要常量级的时间。
		因此，该算法的时间复杂度总共为 O(N)。
		
		自己分析其他问题以提高分析能力。别忘了考虑不同的条件。如果很难对所有情况进行分析，
		请考虑最糟糕的情况。
*/

/* 
========================== 1、环形链表 ==========================
如果链表中有某个节点，可以通过连续跟踪 next 指针再次到达，
则链表中存在环。 为了表示给定链表中的环，我们使用整数 pos 来表
示链表尾连接到链表中的位置（索引从 0 开始）。 如果 pos 是 -1，
则在该链表中没有环。注意：pos 不作为参数进行传递，仅仅是为了标
识链表的实际情况。

如果链表中存在环，则返回 true 。 否则，返回 false 。

进阶：
你能用 O(1)（即，常量）内存解决此问题吗？

示例 1：
输入：head = [3,2,0,-4], pos = 1
输出：true
解释：链表中有一个环，其尾部连接到第二个节点

示例 2：
输入：head = [1,2], pos = 0
输出：true
解释：链表中有一个环，其尾部连接到第一个节点。

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/linked-list/jbex5/
*/

type ListNode struct {
	Val int
	Next *ListNode
}

/* 
	方法一：哈希表法
	思路：
		遍历所有节点，每次遍历到一个节点时，判断该节点此前是否被访问过。
		使用哈希表来存储所有已经访问过的节点。每次我们到达一个节点，
		如果该节点已经存在于哈希表中，则说明该链表是环形链表，
		否则就将该节点加入哈希表中。重复这一过程，直到我们遍历完整个链表即可。
	时间复杂度：O(N)
		其中 N 是链表中的节点数。最坏情况下我们需要遍历每个节点一次。
	空间复杂度：O(N)
		其中 N 是链表中的节点数。主要为哈希表的开销，
		最坏情况下我们需要将每个节点插入到哈希表中一次。
*/
func hasCycle1(head *ListNode) bool {
	if head == nil || head.Next == nil {
		return false
	}
	visited := make(map[*ListNode]bool, 0)
	for head != nil {
		if _, ok := visited[head]; ok {
			return true
		}
		visited[head] = true
		head = head.Next
	}
	return false
}

/* 
	方法二：双指针
	思路：使用快慢双指针，快指针每次走2个单位，慢指针每次走一个单位：
			如果没有环，快指针将停在链表的末尾。
			如果有环，快指针最终将与慢指针相遇。
		每一次迭代，快速指针将额外移动一步。如果环的长度为 M，
		经过 M 次迭代后，快指针肯定会多绕环一周，并赶上慢指针。
	
	时间复杂度：O(N)
		其中 N 是链表中的节点数。
		当链表中不存在环时，快指针将先于慢指针到达链表尾部，
		链表中每个节点至多被访问两次。当链表中存在环时，每一轮移动后，
		快慢指针的距离将减小一。而初始距离为环的长度，因此至多移动 N 轮。
	空间复杂度：O(1)
		我们只使用了两个指针的额外空间。
*/
func hasCycle2(head *ListNode) bool {
    if head == nil || head.Next == nil {
		return false
	}
	// 假设 slow 和 fast 都从虚拟的前置节点出发，走一次后就会如下
	slow, fast := head, head.Next
	// 如果链表有环，快指针必定会追上慢指针
	for slow != fast {
		// fast 能走到链表末尾，说明没环
		if fast == nil || fast.Next == nil {
			return false
		}
		slow = slow.Next
		fast = fast.Next.Next
	}
	return true
}

/* 
========================== 2、环形链表 II==========================
给定一个链表，返回链表开始入环的第一个节点。 如果链表无环，则返回 null。
为了表示给定链表中的环，我们使用整数 pos 来表示链表尾连接到链表中的位置
（索引从 0 开始）。 如果 pos 是 -1，则在该链表中没有环。注意，pos 仅仅
是用于标识环的情况，并不会作为参数传递到函数中。

说明：不允许修改给定的链表。

进阶：
	你是否可以使用 O(1) 空间解决此题？
	
示例 1：
输入：head = [3,2,0,-4], pos = 1
输出：返回索引为 1 的链表节点
解释：链表中有一个环，其尾部连接到第二个节点。

示例 2：
输入：head = [1,2], pos = 0
输出：返回索引为 0 的链表节点
解释：链表中有一个环，其尾部连接到第一个节点。

示例 3：
输入：head = [1], pos = -1
输出：返回 null
解释：链表中没有环。

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/linked-list/jjhf6/
*/
/* 
	方法一：哈希表法
	思路：
		遍历链表中的每个节点，并将它记录下来；一旦遇到了此前遍历过的节点，
		就可以判定链表中存在环。
	时间复杂度：O(N)
		其中 N 为链表中节点的数目。我们恰好需要访问链表中的每一个节点。
	空间复杂度：O(N)
		其中 N 为链表中节点的数目。我们需要将链表中的每个节点都保存在
		哈希表当中。
*/
func detectCycle1(head *ListNode) *ListNode {
    if head == nil {
		return nil
	}
	visited := make(map[*ListNode]bool, 0)
	for head != nil {
		// 返回第一个重复访问的节点，既是环的入口
		if _, ok := visited[head]; ok {
			return head
		}
		visited[head] = true
		head = head.Next
	}
	return nil
}

/* 
	方法二：双指针
	思路：
		我们使用两个指针，fast 与 slow。它们起始都位于链表的头部。
		随后，slow 指针每次向后移动一个位置，而 fast 指针向后移动两个位置。
		如果链表中存在环，则 fast 指针最终将再次与 slow 指针在环中相遇。
		如下图所示，设链表中环外部分的长度为 a。slow 指针进入环后，
		又走了 b 的距离与 fast 相遇。此时，fast 指针已经走完了环的 n 圈，
		因此它走过的总距离为 a+n(b+c)+b=a+(n+1)b+nc。
		         ___bc
		       b/   ·\
		a—————a      |
			   c\___/
		aa的距离为：a，bb的距离为：b，cc的距离为：c
		
		根据题意，任意时刻，fast 指针走过的距离都为 slow 指针的 2 倍。因此，
		我们有：a+(n+1)b+nc=2(a+b) ⟹  a=c+(n−1)(b+c)
			之所以是 2(a+b)，是因为 fast 的速度是 slow 的2倍，在 slow 入环后，
			fast 必然可以在 slow 走完一圈之前追上它，而它们第一次相遇的点就
			是 b 点。
		有了 a=c+(n−1)(b+c) 的等量关系，我们会发现：
			从相遇点到入环点的距离加上 n−1 圈的环长，恰好等于从链表头部到入环
			点的距离。因此，当发现 slow 与 fast 相遇时，我们再额外使用一个
			指针 ptr, 起始，它指向链表头部；随后，它和 slow 每次向后移动一个
			位置，最终，它们会在入环点相遇。
*/
func detectCycle2(head *ListNode) *ListNode {
    if head == nil || head.Next == nil {
		return nil
	}
	slow, fast := head, head
	for fast != nil && fast.Next != nil {
		slow = slow.Next
		fast = fast.Next.Next
		// 第一次相遇后跳出
		if slow == fast {
			break
		}
	}
	// 没有环
	if fast != slow {
		return nil
	}
	// 此后 fast 指向链表头部，用与 slow 相同的速度行走
	fast = head
	// fast、slow 再次相遇就在入环点
	for fast != slow {
		fast = fast.Next
		slow = slow.Next
	}
	return fast
}

/* 
========================== 3、相交链表 ==========================
编写一个程序，找到两个单链表相交的起始节点。
如下面的两个链表：
		a1-a2
			\
			 c1--c2-c3
			/
	b1-b2-b3
在节点 c1 开始相交。

示例1：
		4-1
		   \
			8-4-5
		   /
	  5-0-1
输入：
	intersectVal = 8, listA = [4,1,8,4,5], listB = [5,0,1,8,4,5], skipA = 2, skipB = 3
输出：
	Reference of the node with value = 8
输入解释：
	相交节点的值为 8 （注意，如果两个链表相交则不能为 0）。从各自的表头开始算起，
	链表 A 为 [4,1,8,4,5]，链表 B 为 [5,0,1,8,4,5]。在 A 中，
	相交节点前有 2 个节点；在 B 中，相交节点前有 3 个节点。

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/intersection-of-two-linked-lists
*/
/* 
	方法一：哈希表法
	思路:
		遍历链表 A 并将每个结点的地址/引用存储在哈希表中。
		然后检查链表 B 中的每一个结点 bi​ 是否在哈希表中。
		若在，则 bi​ 为相交结点。
    时间复杂度 : O(m+n)
    空间复杂度 : O(m) 或 O(n)
*/
func getIntersectionNode1(headA, headB *ListNode) *ListNode {
	if headA == nil || headB == nil {
		return nil
	}
	visitedA := make(map[*ListNode]bool, 0)
	for headA != nil {
		visitedA[headA] = true
		headA = headA.Next
	}
	for headB != nil {
		if _, ok := visitedA[headB]; ok {
			return headB
		}
		headB = headB.Next
	}
	return nil
}

/* 
	方法二：双指针
	思路：
		创建两个指针 pA和 pB，分别初始化为链表 A 和 B 的头结点。
		然后让它们向后逐结点遍历。
		当 pA到达链表的尾部时，将它重定位到链表 B 的头结点 (你没看错，
		就是链表 B); 类似的，当 pB 到达链表的尾部时，将它重定位到链表 A 
		的头结点。若在某一时刻 pA和 pB 相遇，则 pA/pB 为相交结点。

		想弄清楚为什么这样可行, 可以考虑以下两个链表: A={1,3,5,7,9,11} 
		和 B={2,4,9,11}，相交于结点 9。 由于 B.length (=4) < A.length (=6)，
		pB 比 pA 少经过 2 个结点，会先到达尾部。将 pB 重定向到 A 的头结点，
		pA重定向到 B 的头结点后，pB 要比 pA多走 2 个结点。因此，它们会同时
		到达交点。
		
		如果两个链表存在相交，它们末尾的结点必然相同。因此当 pA/pB 到达链表
		结尾时，记录下链表 A/B 对应的元素。若最后元素不相同，则两个链表不相交。
		
		可以理解成两个人速度一致， 走过的路程一致。那么肯定会同一个时间点到达
		终点（把相交点看做是终点）。如果到达终点的最后一段路两人都走的话，
		那么这段路上俩人肯定是肩并肩手牵手的。
	
	时间复杂度 : O(m+n)
			m、n 分别表示链表 A、B的节点个数
    空间复杂度 : O(1)
*/
func getIntersectionNode2(headA, headB *ListNode) *ListNode {
	if headA == nil || headB == nil {
		return nil
	}
	pA, pB := headA, headB
	for pA != pB {
		if pA == nil {
			pA = headB
		} else {
			pA = pA.Next
		}
		if pB == nil {
			pB = headA
		} else {
			pB = pB.Next
		}
	}
	// 如果不相交，最后 pA 是 null, pB 是 null, 两个相等，退出循环。
	return pA
}

/* 
========================== 4、删除链表的倒数第N个节点 ==========================
给你一个链表，删除链表的倒数第 n 个结点，并且返回链表的头结点。
进阶：你能尝试使用一趟扫描实现吗？

示例1：
	1-2-3-4-5
输入：head = [1,2,3,4,5], n = 2
输出：[1,2,3,5]

示例 2：
输入：head = [1], n = 1
输出：[]

示例 3：
输入：head = [1,2], n = 1
输出：[1]

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/linked-list/jf1cc/
*/
/* 
	方法一：双指针
	思路：
		定义两个指针p1、p2，p1先走 n 个节点后，p2再以相同的速度
		从表头开始随 p1 一起走，此时 p1、p2 相距 n-1 个节点，
		当 p1 走到链表末尾时（p1= nil），p2 指向的就是链表倒数
		第 n 个节点，考虑到要方便删除操作，我们可以让 p2 指向 head
		的前置节点，这样当 p1 到达链表末尾时，p2 指向的是倒数第 n 个
		节点的上一个节点，我们就可以用 p2 来删除了。
	时间复杂度：O(n)
		我们只需一次遍历链表就可以完成删除操作
	空间复杂度：O(1)
*/
func removeNthFromEnd(head *ListNode, n int) *ListNode {
	pre := &ListNode{}
	pre.Next = head
	p1, p2 := head, pre
	// p1 先走 n 个节点
	for i := 0; i < n; i ++ {
		p1 = p1.Next
	}
	for p1 != nil {
		p1 = p1.Next
		p2 = p2.Next
	}
	// 删除倒数第 n 个节点
	p2.Next = p2.Next.Next
	return pre.Next
}

/* 
========================== 5、回文链表 ==========================
请判断一个链表是否为回文链表。

示例 1:
输入: 1->2
输出: false

示例 2:
输入: 1->2->2->1
输出: true

进阶：
你能否用 O(n) 时间复杂度和 O(1) 空间复杂度解决此题？

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/palindrome-linked-list
*/
/* 
	方法一：快慢指针法寻找链表中点
	思路：
		找到链表中点后，从中点处断开得到两个链表，然后反转第一个链表，
		用中心扩散法对比两个链表的值判断是否是回文
	时间复杂度：O(n)
		n 是链表节点个数
	空间复杂度：O(1)
*/
func isPalindrome(head *ListNode) bool {
	if head == nil || head.Next == nil {
		return true
	}
	// 获取链表的中间节点
	mid := getMid(head)

	// 从中点处断开得到前后两个链表
	list1, list2 := head, mid.Next
	mid.Next = nil

	// 反转第二个链表，反转第一个链表不行
	// 因为当原链表的节点个数为奇数时，第一个链表会比第二个链表多出一个节点，
	// 此节点是原链表的中间节点，此时反转第一个链表会导致将原链表的
	// 中间节点（反转后是第一个链表的头节点）与第二个链表的头结点做对比，
	// 进而导致回文判断出错
	list2 = reverse(list2)

	// 判断是否是回文
	// 注意：链表断开时，如果链表节点个数是奇数，则前一个链表会比后一个
	// 链表多一个节点，判断回文时我们只需以后一个链表的节点数为标准就行了
	for list1 != nil && list2 != nil {
		if list1.Val != list2.Val {
			return false
		}
		list2 = list2.Next
		list1 = list1.Next
	}
	return true
}
func reverse(head *ListNode) *ListNode {
	if head == nil || head.Next == nil {
		return head
	}
	var pre, next *ListNode
	cur := head
	for cur != nil {
		next = cur.Next
		cur.Next = pre
		pre = cur
		cur = next
	}
	return pre
}
/* 
使用快慢指针法寻找链表的中间节点
当链表节点个数为奇数时，返回最中间的那个节点，
是偶数时没有最中间节点，返回最中间两个节点的第一个节点
*/
func getMid(head *ListNode) *ListNode {
	if head == nil || head.Next == nil {
		return head
	}
	slow, fast := head, head.Next
	for fast != nil && fast.Next != nil {
		slow = slow.Next
		fast = fast.Next.Next
	}
	return slow
}


/* 
========================== 6、合并两个有序链表 ==========================
将两个升序链表合并为一个新的 升序 链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的。 
1->2->4
1->3->4		=> 1->1->2->3->4->4

示例 1：

输入：l1 = [1,2,4], l2 = [1,3,4]
输出：[1,1,2,3,4,4]

示例 2：

输入：l1 = [], l2 = []
输出：[]

示例 3：

输入：l1 = [], l2 = [0]
输出：[0]

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/merge-two-sorted-lists
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
*/
/* 
	方法：双指针
	思路：
		使用两个指针遍历两个链表，同时比较两个指针指向的 节点 的值，
		谁小先合并谁，最后再合并剩余节点。
	时间复杂度：O(m+n)
		m、n 分别是 l1、l2 节点的个数
	空间复杂度：O(1)
*/
func mergeTwoLists(l1 *ListNode, l2 *ListNode) *ListNode {
	if l1 == nil && l2 == nil {
		return nil
	}
	if l1 == nil {
		return l2
	}
	if l2 == nil {
		return l1
	}
	head := &ListNode{}
	p := head
	for l1 != nil && l2 != nil {
		if l1.Val < l2.Val {
			p.Next = l1
			l1 = l1.Next
		} else {
			p.Next = l2
			l2 = l2.Next
		}
		p = p.Next
	}
	if l1 != nil {
		p.Next = l1
	} else {
		p.Next = l2
	}
	return head.Next
}

/* 
========================== 7、两数相加 ==========================
给你两个 非空 的链表，表示两个非负的整数。它们每位数字都是按照 逆序 的
方式存储的，并且每个节点只能存储 一位 数字。
请你将两个数相加，并以相同形式返回一个表示和的链表。
你可以假设除了数字 0 之外，这两个数都不会以 0 开头。


示例 1：

l1：2->4->3
l2：5->6->4
返回：7->0->8

输入：l1 = [2,4,3], l2 = [5,6,4]
输出：[7,0,8]
解释：342 + 465 = 807.

示例 2：
输入：l1 = [0], l2 = [0]
输出：[0]

示例 3：
输入：l1 = [9,9,9,9,9,9,9], l2 = [9,9,9,9]
输出：[8,9,9,9,0,0,0,1]
 

提示：
    每个链表中的节点数在范围 [1, 100] 内
    0 <= Node.val <= 9
    题目数据保证列表表示的数字不含前导零

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/linked-list/fv6w7/
*/
/* 
	方法一：模拟加法
	思路：
		模拟加法运算，从个位开始遍历两个链表，对每一位的值进行加法运算
		（l1 节点的值 + l2 节点的值 + 上一位的进位值）获取当前位的值，
		用该值作为节点接入结果链表
	时间复杂度：O(max(m,n))
		m、n是两个链表的节点个数，我们同时处理两个链表，并处理较长链表的
		后续节点和最高进位，所以时间复杂度是量链表中较长的那个
	空间复杂度：O(max(m,n))
		如果最高位的进位 > 0，我们还需要额外的一个节点来返回结果，所以
		我们最多需要的空间是较长链表的节点数+1
*/
func addTwoNumbers(l1 *ListNode, l2 *ListNode) *ListNode {
	if l1 == nil && l2 == nil {
		return nil
	}
	if l1 == nil {
		return l2
	}
	if l2 == nil {
		return l1
	}

	var sum int		// 当前位的和
	var value int	// 当前位的值
	var carry int	// 进位的值
	var res *ListNode = &ListNode{}
	p := res
	// 从个位开始处理两个链表，对较长链表的剩余节点和最高位的进位也要处理
	for l1 != nil || l2 != nil || carry > 0 {
		sum = 0
		if l1 != nil {
			sum += l1.Val
		}
		if l2 != nil {
			sum += l2.Val
		}
		sum += carry
		value = sum % 10
		carry = sum / 10
		p.Next = &ListNode{Val:value}
		p = p.Next
		if l1 != nil {
			l1 = l1.Next
		}
		if l2 != nil {
			l2 = l2.Next
		}
	}
	return res.Next
}

/* 
========================== 8、旋转链表 ==========================
给定一个链表，旋转链表，将链表每个节点向右移动 k 个位置，其中 k 是非负数。

示例 1:
输入: 1->2->3->4->5->NULL, k = 2
输出: 4->5->1->2->3->NULL
解释:
向右旋转 1 步: 5->1->2->3->4->NULL
向右旋转 2 步: 4->5->1->2->3->NULL

示例 2:
输入: 0->1->2->NULL, k = 4
输出: 2->0->1->NULL
解释:
向右旋转 1 步: 2->0->1->NULL
向右旋转 2 步: 1->2->0->NULL
向右旋转 3 步: 0->1->2->NULL
向右旋转 4 步: 2->0->1->NULL

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/rotate-list
*/
/* 
	方法一：尾结点旋转【超出时限】
	思路：
		对链表进行 k 次尾结点旋转，每次旋转时先寻找链表的最后一个节点，
		把该节点旋转到链表头部形成新的链表结构后返回
	时间复杂度：O(nk)
		n 是链表节点个数，k 是需要执行的尾结点旋转次数，我们需要旋转 k 次，
		每次需要先从链表找到最后一个节点（耗时O(n)）再旋转，总耗时O(nk)。
	空间复杂度：O(1)
*/
func rotateRight(head *ListNode, k int) *ListNode {
	if head == nil || head.Next == nil || k == 0 {
		return head
	}
	var rotate func(head *ListNode) *ListNode
	rotate = func(head *ListNode) *ListNode {
		if head == nil {
			return nil
		}
		dummy := &ListNode{}
		dummy.Next = head
		preTail, tail := dummy, head
		for tail.Next != nil {
			tail = tail.Next
			preTail = preTail.Next
		}
		preTail.Next = nil
		tail.Next = head
		head = tail
		return head
	}
	// 执行 k 次尾结点旋转
	for i := 0; i < k; i ++ {
		head = rotate(head)
	}
	return head
}

/* 
	方法二：尾结点旋转【改进】【通过】
	思路：
		前面的方法超时是因为我们执行了 k 次尾结点旋转，而实际上如果
		k == 链表长度的话，k 次旋转前和 k 次旋转后的链表完全一样，
		可以不用旋转，故我们只需预先得到链表长度 n，再执行 k % n 次
		尾结点旋转即可。
	时间复杂度：O(n(k%n))
	空间复杂度：O(1)
*/
func rotateRight(head *ListNode, k int) *ListNode {
	if head == nil || head.Next == nil || k == 0 {
		return head
	}
	var rotate func(head *ListNode) *ListNode
	rotate = func(head *ListNode) *ListNode {
		if head == nil {
			return nil
		}
		dummy := &ListNode{}
		dummy.Next = head
		preTail, tail := dummy, head
		for tail.Next != nil {
			tail = tail.Next
			preTail = preTail.Next
		}
		preTail.Next = nil
		tail.Next = head
		head = tail
		return head
	}
	// 获取链表长度
	n := 0
	p := head
	for p != nil {
		n ++
		p = p.Next
	}
	// 执行 k % n 次尾结点旋转
	for i := 0; i < k % n; i ++ {
		head = rotate(head)
	}
	return head
}

/* 
	方法三：连成环
	思路：
		遍历链表获取链表长度，并将量表尾部与头部相连得到一个环，然后找
		到相应的位置断开这个环，确定新的链表头和链表尾。
		确定新的表头：
			新的表头就在新的表尾的后面，新的表尾在 n - (k % n) - 1 处，
			即新的表头在 n - (k % n) 处
		最后需要断开表尾与表头的连接

		图示：
			输入：
				1->2->3->4->5->nil
				k = 2
			闭环：
				1->2->3->4->5->连到1
				n = 5
			寻找表尾：
					新尾 新头
					  ↓  ↓
				1->2->3->4->5->连到1
			断开环：
				4->5->1->2->3->nil
			输出：
				4->5->1->2->3->nil
	时间复杂度：O(n)
	空间复杂度：O(1)
*/
func rotateRight(head *ListNode, k int) *ListNode {
	if head == nil || head.Next == nil || k == 0 {
		return head
	}
	n := 1
	p := head
	for p.Next != nil {
		n ++
		p = p.Next
	}
	if k % n == 0 {
		return head
	}
	// 连成环
	p.Next = head

	// 寻找新的尾部
	newTail := head
	for i := 0; i < n - (k % n) - 1; i ++ {
		newTail = newTail.Next
	}
	// 获取新的头部
	newHead := newTail.Next
	// 断开环
	newTail.Next = nil
	return newHead
}