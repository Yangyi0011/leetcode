package binarytree

/* 
	二叉树的其他问题
*/

/* 
========================== 1、从中序与后序遍历序列构造二叉树 ==========================
根据一棵树的中序遍历与后序遍历构造二叉树。

注意:
你可以假设树中没有重复的元素。

例如，给出

中序遍历 inorder = [9,3,15,20,7]
后序遍历 postorder = [9,15,7,20,3]

返回如下的二叉树：

    3
   / \
  9  20
    /  \
   15   7

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/data-structure-binary-tree/xo98qt/
*/
/* 
	方法一：递归
	思路：
		首先解决这道题我们需要明确给定一棵二叉树，我们是如何对其进行中序遍历与后序遍历的：
    		中序遍历的顺序是每次遍历左孩子，再遍历根节点，最后遍历右孩子。
    		后序遍历的顺序是每次遍历左孩子，再遍历右孩子，最后遍历根节点。
		因此根据上文所述，我们可以发现后序遍历的数组最后一个元素代表的即为根节点。
		知道这个性质后，我们可以利用已知的根节点信息在中序遍历的数组中找到根节点所在
		的下标，然后根据其将中序遍历的数组分成左右两部分，「左边部分即左子树，右边部分为
		右子树」，针对每个部分可以用同样的方法继续递归下去构造。
	算法：
		为了高效查找根节点元素在中序遍历数组中的下标，我们选择创建哈希表来存储中序序列，
		即建立一个（元素，下标）键值对的哈希表。
		定义递归函数 helper(in_left, in_right) 表示当前递归到中序序列中当前子树的左右
		边界，递归入口为 helper(0, n - 1) ：
		如果 in_left > in_right，说明子树为空，返回空节点。
		选择后序遍历的最后一个节点作为根节点。
		利用哈希表 O(1) 查询当根节点在中序遍历中下标为 index。从 in_left 到 
		index - 1 属于左子树，从 index + 1 到 in_right 属于右子树。
		根据后序遍历逻辑，递归创建右子树 helper(index + 1, in_right) 和左子树 
		helper(in_left, index - 1)。
		
		注意这里有需要先创建右子树，再创建左子树的依赖关系。
		可以理解为在后序遍历的数组中整个数组是先存储左子树的节点，再存储右子树的节点，
		最后存储根节点，如果按每次选择「后序遍历的最后一个节点」为根节点，则先被构造出来
		的应该为右子树。
		返回根节点 root。
	时间复杂度：O(n)
		其中 n 是树中的节点个数。
	空间复杂度：O(n)
		我们需要使用 O(n) 的空间存储哈希表，以及 O(h)（其中 h 是树的高度）的空间表示
		递归时栈空间。这里 h<n，所以总空间复杂度为 O(n)。
*/
func buildTree(inorder []int, postorder []int) *TreeNode {
    idxMap := map[int]int{}
    for i, v := range inorder {
        idxMap[v] = i
    }
    var build func(int, int) *TreeNode
    build = func(inorderLeft, inorderRight int) *TreeNode {
        // 无剩余节点
        if inorderLeft > inorderRight {
            return nil
        }

        // 后序遍历的末尾元素即为当前子树的根节点
        val := postorder[len(postorder)-1]
        postorder = postorder[:len(postorder)-1]
        root := &TreeNode{Val: val}

        // 根据 val 在中序遍历的位置，将中序遍历划分成左右两颗子树
        // 由于我们每次都从后序遍历的末尾取元素，所以要先遍历右子树再遍历左子树
        inorderRootIndex := idxMap[val]
        root.Right = build(inorderRootIndex+1, inorderRight)
        root.Left = build(inorderLeft, inorderRootIndex-1)
        return root
    }
    return build(0, len(inorder)-1)
}

/* 
========================== 2、从前序与中序遍历序列构造二叉树 ==========================
根据一棵树的前序遍历与中序遍历构造二叉树。

注意:
你可以假设树中没有重复的元素。

例如，给出
前序遍历 preorder = [3,9,20,15,7]
中序遍历 inorder = [9,3,15,20,7]

返回如下的二叉树：
    3
   / \
  9  20
    /  \
   15   7

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal
*/
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
func buildTree(preorder []int, inorder []int) *TreeNode {
	idxMap := make(map[int]int, len(inorder))
	for i, v := range inorder {
		idxMap[v] = i
	}
	var build func(int, int) *TreeNode
	build = func(inorderLeft, inorderRight int) *TreeNode {
		// 无剩余节点
		if inorderLeft > inorderRight {
			return nil
		}
		// 从先序数组中取根节点
		// 先序遍历数组的第一个元素即为当前子树的根节点
		value := preorder[0]
		preorder = preorder[1:]
		// 找到根节点在中序数组中的下标
		index := idxMap[value]
		// 根据 val 在中序遍历的下标，将中序遍历数组划分成左右两颗子树
		root := &TreeNode{Val:value}
		root.Left = build(inorderLeft, index - 1)
		root.Right = build(index + 1, inorderRight)
		return root
	}
	return build(0, len(preorder) - 1)
}

/* 
========================== 3、填充每个节点的下一个右侧节点指针 ==========================
给定一个 完美二叉树 ，其所有叶子节点都在同一层，每个父节点都有两个子节点。
二叉树定义如下：
struct Node {
  int val;
  Node *left;
  Node *right;
  Node *next;
}

填充它的每个 next 指针，让这个指针指向其下一个右侧节点。如果找不到下一个右侧
节点，则将 next 指针设置为 NULL。
初始状态下，所有 next 指针都被设置为 NULL。

进阶：
    你只能使用常量级额外空间。
	使用递归解题也符合要求，本题中递归程序占用的栈空间不算做额外的空间复杂度。
	
示例：
		  1
		/   \
	   2     3
	 /  \   /  \
	4    5 6    7
填充
		  1-->
		/   \
	   2---> 3-->
	 /  \   /  \
	4--> 5->6-->7-->

输入：root = [1,2,3,4,5,6,7]
输出：[1,#,2,3,#,4,5,6,7,#]
解释：
	给定二叉树如图 A 所示，你的函数应该填充它的每个 next 指针，
	以指向其下一个右侧节点，如图 B 所示。序列化的输出按层序遍历排列，
	同一层节点由 next 指针连接，'#' 标志着每一层的结束。

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/data-structure-binary-tree/xoo0ts/
*/
/**
 * Definition for a Node.
 * type Node struct {
 *     Val int
 *     Left *Node
 *     Right *Node
 *     Next *Node
 * }
 */

/* 
	方法一：BFS层级遍历
	思路：
		层级遍历，获取每一层级的元素，对这些元素进行指针处理
	时间复杂度：O(n)
	空间复杂度：O(n)
*/
func connect(root *Node) *Node {
	if root == nil {
		return root
	}
	queue := []*Node{root}
	for len(queue) > 0 {
		n := len(queue)
		for i := 0; i < n; i ++ {
			// 每一层的最后一个元素不需要处理
			if i != n - 1 {
				queue[i].Next = queue[i+1]
			}
			if queue[i].Left != nil {
				queue = append(queue, queue[i].Left)
			}
			if queue[i].Right != nil {
				queue = append(queue, queue[i].Right)
			}
		}
		queue = queue[n:]
	}
	return root
}
/* 
	方法二：利用 Next 指针
	思路：
		一棵树中，存在两种类型的 next 指针。
		1、第一种情况是连接同一个父节点的两个子节点。它们可以通过同一个节点直接访问到，
			因此执行下面操作即可完成连接。
				node.left.next = node.right
		2、第二种情况在不同父亲的子节点之间建立连接，这种情况不能直接连接。
			如果每个节点有指向父节点的指针，可以通过该指针找到 next 节点。
			如果不存在该指针，则按照下面思路建立连接：
				第 N 层节点之间建立 next 指针后，再建立第 N+1 层节点的 next 指针。
				可以通过 next 指针访问同一层的所有节点，因此可以使用第 N 层的 next 
				指针，为第 N+1 层节点建立 next 指针。
		具体如下：
			1、从根节点开始，由于第 0 层只有一个节点，所以不需要连接，
				直接为第 1 层节点建立 next 指针即可。该算法中需要注意的一点是，
				当我们为第 N 层节点建立 next 指针时，处于第 N−1 层。当第 N 层
				节点的 next 指针全部建立完成后，移至第 N 层，建立第 N+1 层节点
				的 next 指针。
			2、遍历某一层的节点时，这层节点的 next 指针已经建立。因此我们只需要
				知道这一层的最左节点，就可以按照链表方式遍历，不需要使用队列。
	时间复杂度：O(n)
		n 表示树的节点个数，每个节点只访问一次
	空间复杂度：O(1)
		不需要存储额外的节点。
*/
func connect(root *Node) *Node {
	if root == nil {
		return root
	}
	// 从根节点开始
	leftmost := root
	for leftmost.Left != nil {
		// 遍历当前层级节点组织成的链表，为下一层的节点更新 next 指针
		head := leftmost
		for head != nil {
			// 同子树上的左右节点连接
			head.Left.Next = head.Right
			if head.Next != nil {
				// 跨子树上的节点连接
				head.Right.Next = head.Next.Left
			}
			// 处理当前层级的下一个节点
			head = head.Next
		}
		// 处理下一层级的节点
		leftmost = leftmost.Left
	}
	return root
}

/* 
========================== 4、填充每个节点的下一个右侧节点指针 II ==========================
给定一个二叉树
struct Node {
  int val;
  Node *left;
  Node *right;
  Node *next;
}

填充它的每个 next 指针，让这个指针指向其下一个右侧节点。如果找不到下一个右
侧节点，则将 next 指针设置为 NULL。
初始状态下，所有 next 指针都被设置为 NULL。

进阶：
    你只能使用常量级额外空间。
    使用递归解题也符合要求，本题中递归程序占用的栈空间不算做额外的空间复杂度。

示例：
		  1
		/   \
	   2     3
	 /  \   /  \
	4    5 6    7
填充
		  1-->
		/   \
	   2---> 3-->
	 /  \     \
	4--> 5---->7-->

输入：root = [1,2,3,4,5,null,7]
输出：[1,#,2,3,#,4,5,7,#]
解释：给定二叉树如图 A 所示，你的函数应该填充它的每个 next 指针，以指向其下一个右侧节点，如图 B 所示。序列化输出按层序遍历顺序（由 next 指针连接），'#' 表示每层的末尾。

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/populating-next-right-pointers-in-each-node-ii
*/
/* 
	方法一：层级遍历
	思路：
		本题与【填充每个节点的下一个右侧节点指针】的不同点是本题给的二叉树不是
		完美二叉树，不过这个对我们的层级遍历来说没什么影响，依旧是对每一层的
		元素做处理。
	时间复杂度：O(n)
	空间复杂度：O(n)
*/
func connect(root *Node) *Node {
	if root == nil {
		return root
	}
	queue := []*Node{root}
	for len(queue) > 0 {
		n := len(queue)
		for i := 0; i < n; i ++ {
			if i != n - 1 {
				queue[i].Next = queue[i + 1]
			}
			if queue[i].Left != nil {
				queue = append(queue, queue[i].Left)
			}
			if queue[i].Right != nil {
				queue = append(queue, queue[i].Right)
			}
		}
		queue = queue[n:]
	}
	return root
}

/* 
	方法二：利用 next 指针
	思路：
		在方法一中，因为对树的结构一无所知，所以使用队列保证有序访问同一层的所
		有节点，并建立它们之间的连接。然而不难发现：一旦在某层的节点之间建立了 
		next 指针，那这层节点实际上形成了一个链表。因此，如果先去建立某一层的
		next 指针，再去遍历这一层，就无需再使用队列了。
		基于该想法，提出降低空间复杂度的思路：如果第 i 层节点之间已经建立 
		next 指针，就可以通过 next 指针访问该层的所有节点，同时对于每个第 i 层
		的节点，我们又可以通过它的 left 和 right 指针知道其第 i+1 层的孩子节
		点是什么，所以遍历过程中就能够按顺序为第 i+1 层节点建立 next 指针。
	时间复杂度：O(n)
	空间复杂度：O(1)
*/
func connect(root *Node) *Node {
	if root == nil {
		return root
	}
	// 当前节点，从 root 开始
	cur := root
	for cur != nil {
		// 下一层级的虚拟头节点
		nextLevelHead := &Node{}
		// 下一层级的上一个节点
		nextLevelPre := nextLevelHead
		// 通过当前节点去为下一层级的节点构建 next 指针
		for cur != nil {
			// 先处理下一层级的 left，再处理 right
			if cur.Left != nil {
				nextLevelPre.Next = cur.Left
				nextLevelPre = nextLevelPre.Next
			}
			if cur.Right != nil {
				nextLevelPre.Next = cur.Right
				nextLevelPre = nextLevelPre.Next
			}
			// 处理当前层级的下一个节点
			cur = cur.Next
		}
		// 切换到下一层级，nextLevelPre 已经为我们连接好了下一层级的节点了
		cur = nextLevelHead.Next
	}
	return root
}

/* 
========================== 5、二叉树的最近公共祖先 ==========================
给定一个二叉树, 找到该树中两个指定节点的最近公共祖先。

百度百科中最近公共祖先的定义为：“对于有根树 T 的两个节点 p、q，最近公共祖先表
示为一个节点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（一个节点也可以是它
自己的祖先）。”

示例 1：
          3
       /     \
      5       1
    /  \    /   \
   6    2  0     8
       / \
      7   4
输入：root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 1
输出：3
解释：节点 5 和节点 1 的最近公共祖先是节点 3 。

示例 2：
          3
       /     \
      5       1
    /  \    /   \
   6    2  0     8
       / \
      7   4
输入：root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 4
输出：5
解释：节点 5 和节点 4 的最近公共祖先是节点 5 。因为根据定义最近公共祖先节点可以为节点本身。

示例 3：
    1
   /
  2
输入：root = [1,2], p = 1, q = 2
输出：1

提示：
    树中节点数目在范围 [2, 105] 内。
    -109 <= Node.val <= 109
    所有 Node.val 互不相同 。
    p != q
    p 和 q 均存在于给定的二叉树中。

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/data-structure-binary-tree/xopaih/
*/
/* 
	方法一：递归
	思路：
		我们递归遍历整棵二叉树，定义 fx​ 表示 x 节点的子树中是否包含 p 节点
		或 q 节点，如果包含为 true，否则为 false。那么符合条件的最近公共祖
		先 x 一定满足如下条件：
			(flson && frson) ∣∣ ((x = p ∣∣ x = q) && (flson ∣∣ frson))
		其中 lson和 rson 分别代表 x 节点的左孩子和右孩子。初看可能会感觉条
		件判断有点复杂，我们来一条条看:
			1、(flson && frson)
				flson && frson 说明左子树和右子树均包含 p 节点或 q 节点，
				如果左子树包含的是 p 节点，那么右子树只能包含 q 节点，反之
				亦然，因为 p 节点和 q 节点都是不同且唯一的节点，因此如果满
				足这个判断条件即可说明 x 就是我们要找的最近公共祖先。
			2、((x = p ∣∣ x = q) && (flson ∣∣ frson))
				这个判断条件即是考虑了 x 恰好是 p 节点或 q 节点且它的左子树
				或右子树有一个包含了另一个节点的情况，因此如果满足这个判断条
				件亦可说明 x 就是我们要找的最近公共祖先。

		你可能会疑惑这样找出来的公共祖先深度是否是最大的。其实是最大的，因为
		我们是自底向上从叶子节点开始更新的，所以在所有满足条件的公共祖先中一
		定是深度最大的祖先先被访问到，且由于 fx​ 本身的定义很巧妙，在找到最近
		公共祖先 x 以后，fx​ 按定义被设置为 true ，即假定了这个子树中只有一个
		 p 节点或 q 节点，因此其他公共祖先不会再被判断为符合条件。
	时间复杂度：O(n)
		其中 n 是二叉树的节点数。二叉树的所有节点有且只会被访问一次，因此时
		间复杂度为 O(n)。
	空间复杂度：O(n)
		其中 n 是二叉树的节点数。递归调用的栈深度取决于二叉树的高度，二叉树
		最坏情况下为一条链，此时高度为 n，因此空间复杂度为 O(n)。
*/
func lowestCommonAncestor(root, p, q *TreeNode) *TreeNode {
    if root == nil {
        return nil
	}
	//  ((x = p ∣∣ x = q) && (flson ∣∣ frson))
	// 遇到 p、q 之一就返回
	// 此处处理了 p、q 处在同一子树下的情况，最先遇到的那个即为最近公共祖先
    if root.Val == p.Val || root.Val == q.Val {
        return root
    }
    left := lowestCommonAncestor(root.Left, p, q)
	right := lowestCommonAncestor(root.Right, p, q)
	// (flson && frson)
	// 处理 p、q 分别处在不同子树中的情况，返回其最近公共祖先
    if left != nil && right != nil {
        return root
	}
    if left == nil {
        return right
    }
    return left
}

/* 
	方法二：存储父节点
	思路：
		我们可以用哈希表存储所有节点的父节点，然后我们就可以利用节点的父节点
		信息从 p 结点开始不断往上跳，并记录已经访问过的节点，再从 q 节点开始
		不断往上跳，如果碰到已经访问过的节点，那么这个节点就是我们要找的最近
		公共祖先。

		1、从根节点开始遍历整棵二叉树，用哈希表记录每个节点的父节点指针。
		2、从 p 节点开始不断往它的祖先移动，并用数据结构记录已经访问过的
			祖先节点。
		3、同样，我们再从 q 节点开始不断往它的祖先移动，如果有祖先已经被
			访问过，即意味着这是 p 和 q 的深度最深的公共祖先，即 LCA 节点。
	时间复杂度：O(N)
		其中 N 是二叉树的节点数。二叉树的所有节点有且只会被访问一次，从 p 
		和 q 节点往上跳经过的祖先节点个数不会超过 N，因此总的时间复杂度
		为 O(N)。
	空间复杂度：O(N)
		其中 N 是二叉树的节点数。递归调用的栈深度取决于二叉树的高度，二叉树
		最坏情况下为一条链，此时高度为 N，因此空间复杂度为 O(N)，哈希表存储
		每个节点的父节点也需要 O(N) 的空间复杂度，因此最后总的空间复杂度
		为 O(N)。
*/
func lowestCommonAncestor(root, p, q *TreeNode) *TreeNode {
	// 存储所有节点的父节点
    parent := map[int]*TreeNode{}
    var dfs func(*TreeNode)
    dfs = func(r *TreeNode) {
        if r == nil {
            return
        }
        if r.Left != nil {
            parent[r.Left.Val] = r
            dfs(r.Left)
        }
        if r.Right != nil {
            parent[r.Right.Val] = r
            dfs(r.Right)
        }
    }
	dfs(root)
	// 记录 p 的访问路径
    visited := map[int]bool{}	
    for p != nil {
		visited[p.Val] = true
		// 往父节点上浮
        p = parent[p.Val]
    }
    for q != nil {
		// 第一次遇到 p 走过的路径时返回
        if visited[q.Val] {
            return q
        }
        q = parent[q.Val]
    }
    return nil
}

/* 
========================== 6、二叉树的序列化与反序列化 ==========================
序列化是将一个数据结构或者对象转换为连续的比特位的操作，进而可以将转换后的数据
存储在一个文件或者内存中，同时也可以通过网络传输到另一个计算机环境，采取相反方
式重构得到原数据。

请设计一个算法来实现二叉树的序列化与反序列化。这里不限定你的序列 / 反序列化算
法执行逻辑，你只需要保证一个二叉树可以被序列化为一个字符串并且将这个字符串反序
列化为原始的树结构。

提示: 输入输出格式与 LeetCode 目前使用的方式一致，详情请参阅 LeetCode 序列化
二叉树的格式。你并非必须采取这种方式，你也可以采用其他的方法解决这个问题。

示例 1：
输入：root = [1,2,3,null,null,4,5]
输出：[1,2,3,null,null,4,5]

示例 2：
输入：root = []
输出：[]

示例 3：
输入：root = [1]
输出：[1]

示例 4：
输入：root = [1,2]
输出：[1,2]

提示：
    树中结点数在范围 [0, 104] 内
    -1000 <= Node.val <= 1000

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/serialize-and-deserialize-binary-tree
*/
/* 
	方法一：先序遍历
	思路：
		序列化
			我们可以按照先序遍历的方式把二叉树序列化成字符串，遇到 nil 节点
			的时候输出为 "null"，每个节点之间使用","作为分隔，以便反序列化
			时转为字符数组，空的二叉树直接返回空串。
		反序列化：
			空串直接返回 nil。反序列化时我们需要先将序列化的字符串转为字符数组，
			之后再按先序遍历的方式把字符数组的元素转为树的节点。
*/
type Codec struct {
    
}
func Constructor() Codec {
    return Codec{}
}
// Serializes a tree to a single string.
func (this *Codec) serialize(root *TreeNode) string {
	preOderStr := ""
	if root == nil {
		return preOderStr
	}
	// 先序遍历获得二叉树对应的字符串，空节点用 null 表示
	var DAC func(root *TreeNode) string
	DAC = func(root *TreeNode) string {
		// leaf & nil
		if root == nil {
			return "null,"
		}
		// Divide
		left := DAC(root.Left)
		right := DAC(root.Right)

		// Conquer
		res := strconv.Itoa(root.Val) + ","
		res += left
		res += right
		return res
	}
	preOderStr = DAC(root)
	return preOderStr
}
// Deserializes your encoded data to tree.
func (this *Codec) deserialize(data string) *TreeNode {    
    if data == "" {
		return nil
	}
	arr := strings.Split(data, ",")
	// 对先序遍历字符数组进行先序遍历获得二叉树
	var DFS func() *TreeNode
	DFS = func() *TreeNode {
		// 这里不用担心递归时数组越界问题
		// 因为递归到最底部时，遇到 null 就返回了，不会再继续深入
		if arr[0] == "null" {
			arr = arr[1:]
			return nil
		}
		// 先序遍历，每次处理掉数组第一个元素
		val, _ := strconv.Atoi(arr[0])
		arr = arr[1:]
		node := &TreeNode{Val : val}
		node.Left = DFS()
		node.Right = DFS()
		return node
	}
	return DFS()
}
/**
 * Your Codec object will be instantiated and called as such:
 * ser := Constructor();
 * deser := Constructor();
 * data := ser.serialize(root);
 * ans := deser.deserialize(data);
 */