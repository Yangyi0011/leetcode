package list

/* 
	关于链表的其他问题
*/

/* 
========================== 1、扁平化多级双向链表 ==========================
多级双向链表中，除了指向下一个节点和前一个节点指针之外，
它还有一个子链表指针，可能指向单独的双向链表。这些子列表
也可能会有一个或多个自己的子项，依此类推，生成多级数据结构，
如下面的示例所示。
	1---2---3---4---5---6--NULL
			|
			7---8---9---10--NULL
				|
				11--12--NULL

给你位于列表第一级的头节点，请你扁平化列表，使所有结点出现在单级双链表中。
示例 1：

输入：head = [1,2,3,4,5,6,null,null,null,7,8,9,10,null,null,11,12]
输出：[1,2,3,7,8,11,12,9,10,4,5,6]
解释：
	输入的多级列表如下图所示：
		1---2---3---4---5---6--NULL
				|
				7---8---9---10--NULL
					|
					11--12--NULL
	扁平化后：
		1---2---3---7---8---11---12---9---10---4---5---6--NULL
示例 2：
输入：head = [1,2,null,3]
输出：[1,3,2]
解释：
	输入的多级列表如下图所示：
		1---2---NULL
		|
		3---NULL
	扁平化后：
		1---3---2---NULL
示例 3：
输入：head = []
输出：[]

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/linked-list/fw8v5/
*/
/*
	Definition for a Node.
	type Node struct {
		Val int
		Prev *Node
		Next *Node
		Child *Node
	}
*/
/* 
	方法一：深度优先-递归
	思路：
		将列表顺时针转90度，我们可以将列表视为一颗二叉树，而对列表的扁平化
		处理就是对二叉树进行先序遍历：
		原列表：
			1---2---3---4---5---6--NULL
					|
					7---8---9---10--NULL
						|
						11--12--NULL
		旋转后的列表：
						1
						|
						2
						|
					7---3
					|	|
				11--8	4
				|	|	|
				12	9	5
				|	|	|
			   NULL 10	6
					|	|
				   NULL NULL
		扁平化结果：
			1-2-3-7-8-11-12-9-10-4-5-6-NULL
		实质就是把 child 节点看作是二叉树的 left 节点，next 节点看作是二叉树
		的 right 节点，扁平化处理就是模拟二叉树的先序遍历：
			1、定义递归函数 flatten_dfs(prev, curr)，它接收两个指针作为函
				数参数并返回扁平化列表中的尾部指针。curr 指针指向我们要扁平
				化的子列表，prev 指针指向 curr 指向元素的前一个元素。
			2、在函数 flatten_dfs(prev, curr) 我们首先在 prev 和 curr 节点之间建立双向连接。
			3、然后在函数中调用 flatten_dfs(curr, curr.child) 对左子树
				（curr.child 即子列表）进行操作，它将返回扁平化子列表的尾部元
				素 tail，再调用 flatten_dfs(tail, curr.next) 对右子树进行操作。
			4、为了得到正确结果，需要注意以下细节：
				（1）在调用 flatten_dfs(curr, curr.child) 之前我们应该复制 curr.next 
					指针，因为 curr.next 可能在函数中改变。
				（2）在扁平化 curr.child 指针所指向的列表以后，我们应该删除 child 指针，
					因为我们最终不再需要该指针。
	时间复杂度：O(N)
		N 指的是列表的节点数，深度优先搜索遍历每个节点一次。
	空间复杂度：O(N)
		N 指的是列表的节点数，二叉树很可能不是个平衡的二叉树，若节点仅通
		过 child 指针相互链接，则在递归调用的过程中堆栈的深度会达到 N。
*/
func flatten(root *Node) *Node {
    if root == nil {
		return nil
	}

	var flatten_dfs func(prev, curr *Node) *Node
	flatten_dfs = func(prev, curr *Node) *Node {
		if curr == nil {
			return prev
		}
		// 构建双向连接
		prev.Next = curr
		curr.Prev = prev

		// 记录 next
		next := curr.Next

		// 处理左子树并返回左子树的最后一个节点
		tail := flatten_dfs(curr, curr.Child)
		// 清空当前节点的 Child 指针
		curr.Child = nil

		// 用 tail 作为 next 的上一个节点来处理右子树
		// 并返回右子树的最后一个节点
		return flatten_dfs(tail, next)
	}

	// 创建虚拟头结点
	pre := &Node{}
	pre.Next = root

	// 模拟先序遍历
	flatten_dfs(pre, root)

	// 分开虚拟头结点
	pre.Next.Prev = nil
	return pre.Next
}

/* 
	方法二：深度优先-迭代
	思路：
		与递归一样，只是把递归方式改为迭代方式：
			1、首先我们创建 stack，然后将头节点压栈。利用 prev 变量帮
				助我们记录在每个迭代过程的前继节点。
			2、然后我们进入循环迭代 stack 中的元素，直到栈为空。
			3、在每一次迭代过程中，首先在 stack 弹出一个节点（叫做 curr）。
				再建立 prev 和 curr 之间的双向链接，再顺序处理 curr.next 
				和 curr.child 指针所指向的节点，严格按照此顺序执行。
			4、如果 curr.next 存在（即存在右子树），那么我们将 
				curr.next 压栈后进行下一次迭代。
			5、如果 curr.child 存在（即存在左子树），那么将 curr.child 
				压栈，与 curr.next 不同的是，我们需要删除 curr.child 指针，
				因为在最终的结果不再需要使用它。

	时间复杂度：O(N)
		N 指的是列表的节点数，深度优先搜索遍历每个节点一次。
	空间复杂度：O(N)
		N 指的是列表的节点数，二叉树很可能不是个平衡的二叉树，若节点仅通
		过 child 指针相互链接，则在递归调用的过程中堆栈的深度会达到 N。
*/
func flatten(root *Node) *Node {
	if root == nil {
		return nil
	}
	// 保存链表头部
	dummy := &Node{}
	dummy.Next = root

	pre, curr := dummy, dummy.Next
	stack := []*Node{root}
	for len(stack) > 0 {
		curr = stack[len(stack) - 1]
		stack = stack[: len(stack) - 1]
		pre.Next = curr
		curr.Prev = pre

		if curr.Next != nil {
			stack = append(stack, curr.Next)
		}
		if curr.Child != nil {
			stack = append(stack, curr.Child)
			curr.Child = nil
		}
		pre = curr
	}
	dummy.Next.Prev = nil
	return dummy.Next
}

/* 
========================== 2、复制带随机指针的链表 ==========================
给你一个长度为 n 的链表，每个节点包含一个额外增加的随机指针 random ，该指针可以指向链表中的任何节点或空节点。

构造这个链表的 深拷贝。 深拷贝应该正好由 n 个 全新 节点组成，其中每个新节点的值都设为其对应的原节点的值。新节点的 next 指针和 random 指针也都应指向复制链表中的新节点，并使原链表和复制链表中的这些指针能够表示相同的链表状态。复制链表中的指针都不应指向原链表中的节点 。

例如，如果原链表中有 X 和 Y 两个节点，其中 X.random --> Y 。那么在复制链表中对应的两个节点 x 和 y ，同样有 x.random --> y 。

返回复制链表的头节点。

用一个由 n 个节点组成的链表来表示输入/输出中的链表。每个节点用一个 [val, random_index] 表示：
    val：一个表示 Node.val 的整数。
    random_index：随机指针指向的节点索引（范围从 0 到 n-1）；如果不指向任何节点，则为  null 。

你的代码 只 接受原链表的头节点 head 作为传入参数。
 
示例 1：
输入：head = [[7,null],[13,0],[11,4],[10,2],[1,0]]
输出：[[7,null],[13,0],[11,4],[10,2],[1,0]]

示例 2：
输入：head = [[1,1],[2,1]]
输出：[[1,1],[2,1]]

示例 3：
输入：head = [[3,null],[3,0],[3,null]]
输出：[[3,null],[3,0],[3,null]]

示例 4：
输入：head = []
输出：[]
解释：给定的链表为空（空指针），因此返回 null。

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/copy-list-with-random-pointer
*/
/**
 * Definition for a Node.
 * type Node struct {
 *     Val int
 *     Next *Node
 *     Random *Node
 * }
 */
 func copyRandomList(head *Node) *Node {
	// return copyRandomListByHash(head)
	return copyRandomListByLoop(head)
}

/* 
	方法一，hash表法
	思路：
		使用 hash 表记录链表中的节点，原有节点作为key，新建的复制节点作为value，
		在处理复制节点随机指针的过程中，如果随机指针指向的节点还没有被复制创建，
		则先复制创建，并存放到哈希表中
	时间复杂度：O(n)
		n 为链表节点个数，我们需要遍历复制每一个节点
	空间复杂度：O(n)
		我们需要用哈希表存储每一个节点
*/
func copyRandomListByHash(head *Node) *Node {
	if head == nil {
		return nil
	}
	// 使用 hash 表记录链表中的节点，原有节点作为key，新建的复制节点作为value
	hashTable := make(map[*Node] *Node)
	copy := &Node{}
	cur := head
	pre := copy
	for cur != nil {
		if hashTable[cur] == nil {
			hashTable[cur] = &Node{Val:cur.Val}
		}
		pre.Next = hashTable[cur]
		if cur.Random != nil {
			if hashTable[cur.Random] == nil {
				hashTable[cur.Random] = &Node{Val:cur.Random.Val}
			}
			pre.Next.Random = hashTable[cur.Random]
		}
		cur = cur.Next
		pre = pre.Next
	}
	return copy.Next
}
// 方法二：错位空间迭代
// 时间复杂度：O(n)，空间复杂度：O(1)
/* 
思路：
与上面提到的维护一个旧节点和新节点对应的字典不同，我们通过扭曲原来的链表，并将每个拷贝节点都放在原来对应节点的旁边。
这种旧节点和新节点交错的方法让我们可以在不需要额外空间的情况下解决这个问题。

1、遍历原来的链表并拷贝每一个节点，将拷贝节点放在原来节点的旁边，创造出一个旧节点和新节点交错的链表。
	我们只是用了原来节点的值拷贝出新的节点。原节点 next 指向的都是新创造出来的节点。
		cloned_node.next = original_node.next
		original_node.next = cloned_node
	原链表：A->B->C
	错位拼接后的链表：A->A'->B->B'->C->C'
2、迭代这个新旧节点交错的链表，并用旧节点的 random 指针去更新对应新节点的 random 指针。
	比方说，B 的 random 指针指向 A ，意味着 B' 的 random 指针指向 A' 。
3、现在 random 指针已经被赋值给正确的节点， next 指针也需要被正确赋值，以便将新的节点正确链接同时将旧节点重新正确链接。
	原链表：A->B->C
	新链表：A'->B'->C'
*/
func copyRandomListByLoop(head *Node) *Node {
	if head == nil {
		return nil
	}
	// 复制每一个节点插入到原节点之后
	cur := head
	for cur != nil {
		node := &Node{Val : cur.Val}
		node.Next = cur.Next
		cur.Next = node
		cur = cur.Next.Next
	}
	// 处理随机指针
	// 复制节点的随机指针指向的节点即为原节点的随机指针指向的节点的下一个节点
	cur = head
	for cur != nil {
		if cur.Random != nil {
			cur.Next.Random = cur.Random.Next
		}
		cur = cur.Next.Next
	}
	// 分离原节点与复制节点得到复制链表
	copyList := head.Next
	p := head
	c := copyList
	for p != nil {
		p.Next = p.Next.Next
		if c.Next != nil {
			c.Next = c.Next.Next
		}
		p = p.Next
		c = c.Next
	}
	return copyList
}