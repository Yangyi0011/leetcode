package binarysearchtree

/* 
	二叉搜索树
		二叉搜索树（BST）是二叉树的一种特殊表示形式，它满足如下特性：
			每个节点中的值必须大于（或等于）存储在其左侧子树中的任何值。
			每个节点中的值必须小于（或等于）存储在其右子树中的任何值。
		
				5
			   / \
			  2   6
			 / \   \
			1   4   7
			   /
			  3
		普通的二叉树一样，我们可以按照前序、中序和后序来遍历一个二叉搜索
		树。 但是值得注意的是，对于二叉搜索树，我们可以通过中序遍历得到一
		个递增的有序序列。因此，中序遍历是二叉搜索树中最常用的遍历方法。
*/
/* 
========================== 1、验证二叉搜索树 ==========================
给定一个二叉树，判断其是否是一个有效的二叉搜索树。
假设一个二叉搜索树具有如下特征：
    节点的左子树只包含小于当前节点的数。
    节点的右子树只包含大于当前节点的数。
    所有左子树和右子树自身必须也是二叉搜索树。

示例 1:
输入:
    2
   / \
  1   3
输出: true

示例 2:
输入:
    5
   / \
  1   4
     / \
    3   6
输出: false
解释: 输入为: [5,1,4,null,null,3,6]。
     根节点的值为 5 ，但是其右子节点值为 4 。

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/introduction-to-data-structure-binary-search-tree/xpkc6i/
*/
/* 
	方法一：中序遍历
	思路：
		中序遍历，检查结果列表是否已经有序
	时间复杂度：O(n)
		n 表示二叉树节点个数，我们需要先执行中序遍历保存遍历结果到数组，
		再遍历数组查看是否有序，每一次的时间复杂度都是 O(n)
	空间复杂度：O(n)
		我们需要用一个长度为 n 的数组来存储中序遍历数组，递归中序遍历需要 
		O(logn) 栈空间，总的复杂度是 O(n)
*/
func isValidBST(root *TreeNode) bool {
    result := make([]int, 0)
    inOrder(root, &result)
    // check order
    for i := 0; i < len(result) - 1; i++{
        if result[i] >= result[i+1] {
            return false
        }
    }
    return true
}
func inOrder(root *TreeNode, result *[]int)  {
    if root == nil{
        return
    }
    inOrder(root.Left, result)
    *result = append(*result, root.Val)
    inOrder(root.Right, result)
}

/* 
	方法二：分治法
	思路：
		分治法，判断 左 MAX < 根 < 右 MIN
	时间复杂度：O(n)
		n 表示树的节点个数，我们需要完全验证每一个节点是否符合规则。
	空间复杂度：O(n)，
		其中 n 为二叉树的节点个数。递归函数在递归过程中需要为每一层递归
		函数分配栈空间，所以这里需要额外的空间且该空间取决于递归的深度，
		即二叉树的高度。最坏情况下二叉树为一条链，树的高度为 n ，递归最深
		达到 n 层，故最坏情况下空间复杂度为 O(n) 。
*/
type ResultType struct {
	IsValid bool
    // 记录左右两边最大最小值，和根节点进行比较
	Max     *TreeNode
	Min     *TreeNode
}
func isValidBST2(root *TreeNode) bool {
	result := helper(root)
	return result.IsValid
}
func helper(root *TreeNode) ResultType {
	result := ResultType{}
	// check
	if root == nil {
		result.IsValid = true
		return result
	}

	left := helper(root.Left)
	right := helper(root.Right)

	if !left.IsValid || !right.IsValid {
		result.IsValid = false
		return result
	}
	if left.Max != nil && left.Max.Val >= root.Val {
		result.IsValid = false
		return result
	}
	if right.Min != nil && right.Min.Val <= root.Val {
		result.IsValid = false
		return result
	}

	result.IsValid = true
    // 如果左边还有更小的3，就用更小的节点，不用4
    //  5
    // / \
    // 1   4
    //      / \
    //     3   6
	result.Min = root
	if left.Min != nil {
		result.Min = left.Min
	}
	result.Max = root
	if right.Max != nil {
		result.Max = right.Max
	}
	return result
}

/* 
	方法三：DFS
	思路：
		根据二分查找树的性质：
			节点的左子树只包含小于当前节点的数。
			节点的右子树只包含大于当前节点的数。
			所有左子树和右子树自身必须也是二叉搜索树。
		我们可以定义每一颗子树的上限(upper)和下限(lower)，二叉查找树的每一
		个 node 节点的值必须处于该子树的上限与下限之间，否则不是二分查找树。
			对于每一个左子树的节点，其范围是：(lower, node.Val)
			对于每一个右子树的节点，其范围是：(node.Val, upper)
			根节点的上限和下限我们定义为：(int64.min, int64.max)
	时间复杂度：O(n)
		n 表示树的节点个数，我们需要完全验证每一个节点是否符合规则。
	空间复杂度：O(n)
		其中 n 为二叉树的节点个数。递归函数在递归过程中需要为每一层递归函数
		分配栈空间，所以这里需要额外的空间且该空间取决于递归的深度，即二叉树的高度。
		最坏情况下二叉树为一条链，树的高度为 n ，递归最深达到 n 层，故最坏情况下空
		间复杂度为 O(n) 。
*/
func isValidBST(root *TreeNode) bool {
	return isBST(root, -1 << 63, 1 << 63 - 1)
}
func isBST(root *TreeNode, lower, upper int) bool {
	if root == nil {
		return true
	}
	if root.Val <= lower || root.Val >= upper {
		return false
	}
	left := isBST(root.Left, lower, root.Val)
	right := isBST(root.Right, root.Val, upper)
	return left && right
}

/* 
========================== 2、二叉搜索树迭代器 ==========================
实现一个二叉搜索树迭代器类 BSTIterator ，表示一个按中序遍历二叉搜索树
（BST）的迭代器：
	1、BSTIterator(TreeNode root) 初始化 BSTIterator 类的一个对象。BST 
		的根节点 root 会作为构造函数的一部分给出。指针应初始化为一个不存在
		于 BST 中的数字，且该数字小于 BST 中的任何元素。
	2、boolean hasNext() 如果向指针右侧遍历存在数字，则返回 true ；否则
	返回 false 。
    3、int next() 将指针向右移动，然后返回指针处的数字。

注意，指针初始化为一个不存在于 BST 中的数字，所以对 next() 的首次调用将返
	回 BST 中的最小元素。
你可以假设 next() 调用总是有效的，也就是说，当调用 next() 时，BST 的中序
遍历中至少存在一个下一个数字。

示例：
	  7
	/  \
   3    15
	   /  \
	  9    20
输入
["BSTIterator", "next", "next", "hasNext", "next", "hasNext", "next", "hasNext", "next", "hasNext"]
[[[7, 3, 15, null, null, 9, 20]], [], [], [], [], [], [], [], [], []]
输出
[null, 3, 7, true, 9, true, 15, true, 20, false]

解释
BSTIterator bSTIterator = new BSTIterator([7, 3, 15, null, null, 9, 20]);
bSTIterator.next();    // 返回 3
bSTIterator.next();    // 返回 7
bSTIterator.hasNext(); // 返回 True
bSTIterator.next();    // 返回 9
bSTIterator.hasNext(); // 返回 True
bSTIterator.next();    // 返回 15
bSTIterator.hasNext(); // 返回 True
bSTIterator.next();    // 返回 20
bSTIterator.hasNext(); // 返回 False

提示：
    树中节点的数目在范围 [1, 10^5] 内
    0 <= Node.val <= 10^6
    最多调用 10^5 次 hasNext 和 next 操作

进阶：
	你可以设计一个满足下述条件的解决方案吗？next() 和 hasNext() 操作均
	摊时间复杂度为 O(1) ，并使用 O(h) 内存。其中 h 是树的高度。

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/introduction-to-data-structure-binary-search-tree/xpg4qe/
*/
/**
 * Your BSTIterator object will be instantiated and called as such:
 * obj := Constructor(root);
 * param_1 := obj.Next();
 * param_2 := obj.HasNext();
 */

/* 
	方法一：中序遍历
	思路：
		使用中序遍历保存树的中序数组，再从数组中处理 Next() 和 HasNext()
		操作
	时间复杂度：O(n)
		n 是树的节点个数，中序遍历的时间复杂度是 O(n)， Next 和 
		HasNext 的时间复杂度是 O(1)
	空间复杂度：O(n)
		我们需要一个数组来存储遍历结果，需要 O(n) 的额外空间，且中序遍历
		需要的空间为 O(h)，h 是树的高度，总的空间复杂度是 O(n)。
*/
type BSTIterator struct {
	// 标识当前遍历指针的下标
	currentIndex int
	inOrderArr []int
}

func Constructor(root *TreeNode) BSTIterator {
	arr := make([]int, 0)
	if root == nil {
		return BSTIterator{-1, arr}
	}
	stack := make([]*TreeNode, 0)
	for root != nil || len(stack) > 0 {
		for root != nil {
			stack = append(stack, root)
			root = root.Left
		}
		node := stack[len(stack) - 1]
		stack = stack[: len(stack) - 1]
		arr = append(arr, node.Val)
		root = node.Right
	}
	return BSTIterator{-1, arr}
}

func (this *BSTIterator) Next() int {
	this.currentIndex ++
	return this.inOrderArr[this.currentIndex]
}

func (this *BSTIterator) HasNext() bool {
	return this.currentIndex + 1 < len(this.inOrderArr)
}

/* 
	方法二：中序遍历【迭代】
	思路：
		我们可以利用栈这一数据结构，通过迭代的方式对二叉树做中序遍历。
		此时，我们无需预先计算出中序遍历的全部结果，只需要实时维护当前
		栈的情况即可。
	时间复杂度：O(n)
		初始化和调用 hasNext() 都只需要 O(1) 的时间。每次调用 next() 
		函数最坏情况下需要 O(n) 的时间；但考虑到 n 次调用 next() 
		函数总共会遍历全部的 n 个节点，因此总的时间复杂度为 O(n)，因此
		单次调用平均下来的均摊复杂度为 O(1)。
	空间复杂度：O(n)
		其中 n 是二叉树的节点数量。空间复杂度取决于栈深度，而栈深度在二
		叉树为一条链的情况下会达到 O(n)的级别。
*/
type BSTIterator struct {
	currentNode *TreeNode
	stack []*TreeNode
}
func Constructor(root *TreeNode) BSTIterator {
	return BSTIterator{currentNode: root}
}
func (this *BSTIterator) Next() int {
	for node := this.currentNode; node != nil; node = node.Left {
		this.stack = append(this.stack, node)
	}
	this.currentNode = this.stack[len(this.stack) - 1]
	this.stack = this.stack[: len(this.stack) - 1]
	val := this.currentNode.Val
	this.currentNode = this.currentNode.Right
	return val
}
func (this *BSTIterator) HasNext() bool {
	return this.currentNode != nil || len(this.stack) > 0
}

/* 
========================== 3、二叉搜索树中的搜索 ==========================
给定二叉搜索树（BST）的根节点和一个值。 你需要在BST中找到节点值等于给定
值的节点。 返回以该节点为根的子树。 如果节点不存在，则返回 NULL。
例如，
给定二叉搜索树:

        4
       / \
      2   7
     / \
    1   3

和值: 2

你应该返回如下子树:
      2     
     / \   
    1   3

在上述示例中，如果要找的值是 5，但因为没有节点值为 5，我们应该返回 NULL。

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/introduction-to-data-structure-binary-search-tree/xpsqtv/
*/
/* 
	方法一：递归
	思路：
		根据二叉搜索树的性质，我们可以在树中这样搜索：
			1、目标值 == 当前节点的值，返回当前节点
			2、目标值 < 当前节点的值，往当前节点的左子树找
			3、目标值 > 当前节点的值，往当前节点的右子树找
			4、都找不到返回 nil
	时间复杂度：O(h)
		h 是二叉树的高度, n 是二叉树的节点个数，我们每次只会在左、右子树
		中的一个子树进行搜索，平均复杂度为 O(logn)，最坏情况下数是一条链，
		此时时间复杂度是O(n)。
	空间复杂度：O(h)
		h 是二叉树高度，递归搜索需要消耗 O(h) 的栈空间，平均空间复杂度是
		O(logn)，最坏空间复杂度是 O(n)。
*/
func searchBST(root *TreeNode, val int) *TreeNode {
	if root == nil || val == root.Val {
		return root
	}
	if val < root.Val {
		return searchBST(root.Left, val)
	}
	return searchBST(root.Right, val)
}

/* 
	方法二：迭代
	思路：
		依旧根据二叉搜索树的性质来进行搜索，只是把递归改为迭代。
	时间复杂度：O(h)
		h 是二叉树的高度, n 是二叉树的节点个数，我们每次只会在左、右子树
		中的一个子树进行搜索，平均复杂度为 O(logn)，最坏情况下数是一条链，
		此时时间复杂度是O(n)。
	空间复杂度：O(1)
*/
func searchBST(root *TreeNode, val int) *TreeNode {
	for root != nil && val != root.Val {
		if val < root.Val {
			root = root.Left
		} else {
			root = root.Right
		}
	}
	return root
}

/* 
========================== 3、二叉搜索树中的插入操作 ==========================
给定二叉搜索树（BST）的根节点和要插入树中的值，将值插入二叉搜索树。 返回
插入后二叉搜索树的根节点。 输入数据 保证 ，新值和原始二叉搜索树中的任意
节点值都不同。

注意，可能存在多种有效的插入方式，只要树在插入后仍保持为二叉搜索树即可。
你可以返回 任意有效的结果 
示例 1：
输入：root = [4,2,7,1,3], val = 5
输出：[4,2,7,1,3,5]

示例 2：
输入：root = [40,20,60,10,30,50,70], val = 25
输出：[40,20,60,10,30,50,70,null,null,25]

示例 3：
输入：root = [4,2,7,1,3,null,null,null,null,null,null], val = 5
输出：[4,2,7,1,3,5]

提示：
    给定的树上的节点数介于 0 和 10^4 之间
    每个节点都有一个唯一整数值，取值范围从 0 到 10^8
    -10^8 <= val <= 10^8
    新值和原始二叉搜索树中的任意节点值都不同

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/introduction-to-data-structure-binary-search-tree/xp1llt/
*/

/* 
	方法一：递归插入到叶子节点
	思路：
		先在二叉树中搜索到合适的位置，再进行插入。
	时间复杂度：O(h)
		h 是树的高度，最坏情况下树一条链，此时复杂度为O(n)
	空间复杂度：O(h)
		h 是树的高度，最坏情况下树一条链，此时复杂度为O(n)
*/
func insertIntoBST(root *TreeNode, val int) *TreeNode {
	if root == nil {
		return &TreeNode{Val : val}
	}
	if val < root.Val {
		root.Left = insertIntoBST(root.Left, val)
	} else {
		root.Right = insertIntoBST(root.Right, val)
	}
	return root
}
/* 
	方法二：迭代插入到叶子节点
	思路：
		先在二叉树中搜索到合适的位置，再进行插入。
	时间复杂度：O(h)
		h 是树的高度，最坏情况下树一条链，此时复杂度为O(n)
	空间复杂度：O(1)
*/
func insertIntoBST(root *TreeNode, val int) *TreeNode {
	if root == nil {
		return &TreeNode{Val : val}
	}
	// father 记录当前节点的父节点
	father, cur := root, root
	// 寻找合适的叶子节点位置
	for cur != nil {
		father = cur
		if val < cur.Val {
			cur = cur.Left
		} else {
			cur = cur.Right
		}
	}
	// 进行插入
	if val < father.Val {
		father.Left = &TreeNode{Val : val}
	} else {
		father.Right = &TreeNode{Val : val}
	}
	return root
}

/* 
========================== 4、二叉搜索树中的删除操作 ==========================
给定一个二叉搜索树的根节点 root 和一个值 key，删除二叉搜索树中的 key 对
应的节点，并保证二叉搜索树的性质不变。返回二叉搜索树（有可能被更新）的根
节点的引用。

一般来说，删除节点可分为两个步骤：
    首先找到需要删除的节点；
    如果找到了，删除它。

说明： 要求算法时间复杂度为 O(h)，h 为树的高度。

示例:
root = [5,3,6,2,4,null,7]
key = 3
    5
   / \
  3   6
 / \   \
2   4   7
给定需要删除的节点值是 3，所以我们首先找到 3 这个节点，然后删除它。

一个正确的答案是 [5,4,6,2,null,null,7], 如下图所示。
    5
   / \
  4   6
 /     \
2       7

另一个正确答案是 [5,2,6,null,4,null,7]。
    5
   / \
  2   6
   \   \
    4   7

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/introduction-to-data-structure-binary-search-tree/xpcnds/
*/
/* 
	方法一：DFS递归删除
	思路：
		删除二叉搜索树的节点有三种情况：
			1、被删除节点只有左子树，则将其返回给被删除节点的父节点
			2、被删除节点只有右子树，则将其返回给被删除节点的父节点
			3、若被删除节点同时有左、右子树，则在右子树中找到最左节点（
				即被删除节点的中序后继节点），再把被删除节点的左子树挂
				到中序后继节点的左子树下，返回被删除节点的右子树。
	时间复杂度：O(H)
		H 是树的高度。在算法的执行过程中，我们一直在树上向左或向右移动。
		首先先用 O(H1) 的时间找到要删除的节点，H1​ 值得是从根节点到要删除节点的高度。
		然后删除节点需要 O(H2) 的时间，H2​ 指的是从要删除节点到替换节点的高度。
		由于 O(H1+H2)=O(H)，H 指的是树的高度，若树是一个平衡树则 H = log⁡N。
	空间复杂度：O(H)
		递归时堆栈使用的空间，H 是树的高度。
*/
func deleteNode(root *TreeNode, key int) *TreeNode {
	if root == nil {
		return nil
	}
	// 找到要删除的节点
	if key == root.Val {
		// 没有节点或只有右节点
		if root.Left == nil {
			return root.Right
		}
		// 只有左节点
		if root.Right == nil {
			return root.Left
		}
		// 同时有左右节点
		// 找到被删除节点的中序后继节点（右子树中最左的节点）
		cur := root.Right
		for cur.Left != nil {
			cur = cur.Left
		}
		// 把被删除节点的左子树挂载在它的中序后继节点的左子树下
		cur.Left = root.Left
		// 返回被删除节点的右子树，相当于把 root 节点删除了
		return root.Right
	}
	if key < root.Val {
		root.Left = deleteNode(root.Left, key)
	} else {
		root.Right = deleteNode(root.Right, key)
	}
	return root
}

/* 
========================== 5、数据流中的第K大元素 ==========================
设计一个找到数据流中第 k 大元素的类（class）。注意是排序后的第 k 大元素，
不是第 k 个不同的元素。

请实现 KthLargest 类：
    KthLargest(int k, int[] nums) 使用整数 k 和整数流 nums 初始化对象。
	int add(int val) 将 val 插入数据流 nums 后，返回当前数据流中第 k 大
	的元素。

示例：
输入：
["KthLargest", "add", "add", "add", "add", "add"]
[[3, [4, 5, 8, 2]], [3], [5], [10], [9], [4]]
输出：
[null, 4, 5, 5, 8, 8]

解释：
KthLargest kthLargest = new KthLargest(3, [4, 5, 8, 2]);
kthLargest.add(3);   // return 4
kthLargest.add(5);   // return 5
kthLargest.add(10);  // return 5
kthLargest.add(9);   // return 8
kthLargest.add(4);   // return 8

 
提示：
    1 <= k <= 104
    0 <= nums.length <= 104
    -104 <= nums[i] <= 104
    -104 <= val <= 104
    最多调用 add 方法 104 次
    题目数据保证，在查找第 k 大元素时，数组中至少有 k 个元素

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/introduction-to-data-structure-binary-search-tree/xpjovh/
*/
/**
 * Your KthLargest object will be instantiated and called as such:
 * obj := Constructor(k, nums);
 * param_1 := obj.Add(val);
 */
/* 
	方法一：小顶堆【手动建堆】
	思路：
		构建一个容量为 k 的小顶堆，遍历 nums 把大于堆顶的元素替换到堆顶，
		调整堆结构，最后返回堆顶元素即为第 k 大的数。
	时间复杂度：O(nlogk)
		n 是数据流元素个数，建堆操作耗时 O(k)，每次调整堆结构耗时 O(logk)
		总的时间复杂度是 O(nlogk)
	空间复杂度：O(k)
*/
type KthLargest struct {
	k int
	heap []int
}
// 构建堆
func buidHeap(data []int) {
	n := len(data)
	// 从最后一个非叶子节点开始处理，直到根节点
	lastFatherIndex := n / 2 - 1
	for i := lastFatherIndex; i >= 0; i -- {
		fixHeap(data, i, n)
	}
}
// 调整堆
func fixHeap(data []int, fatherIndex, n int) {
	// i := 2*fatherIndex + 1：从当前父节点的第一个子节点开始处理
	// i = 2*i + 1：处理下一个父节点
	for i := 2*fatherIndex + 1; i < n; i = 2*i + 1 {
		// 找到当前节点的最小子节点下标
		if i + 1 < n && data[i] > data[i + 1] {
			i ++
		}
		// 如果当前节点是最小的，直接跳出循环
		if data[fatherIndex] < data[i] {
			break
		}
		// 否则交换当前节点和较小的子节点
		data[i], data[fatherIndex] = data[fatherIndex], data[i]
		// 【重要】父节点下标下移
		fatherIndex = i
	}
}
func Constructor(k int, nums []int) KthLargest {
	n := len(nums)
	heap := make([]int, 0)
	// 防止越界，此时 len(nums) 有可能是小于 k 的
	for i := 0; i < n && len(heap) < k; i ++ {
		heap = append(heap, nums[i])
	}
	buidHeap(heap)
	// 能进这个循环说明 len(nums) >= k
	for i := len(heap); i < n; i ++ {
		if nums[i] > heap[0] {
			heap[0] = nums[i]
			fixHeap(heap, 0, len(heap))
		}
	}
	return KthLargest{k, heap}
}
func (this *KthLargest) Add(val int) int {
	if len(this.heap) < this.k {
		this.heap = append(this.heap, val)
		fixHeap(this.heap, 0, len(this.heap))
	} else {
		if val > this.heap[0] {
			this.heap[0] = val
			fixHeap(this.heap, 0, len(this.heap))
		}
	}
	return this.heap[0]
}

/* 
	方法二：小顶堆【实现接口】
	思路：
		构建一个容量为 k 的小顶堆，遍历 nums 把大于堆顶的元素替换到堆顶，
		调整堆结构，最后返回堆顶元素即为第 k 大的数。
	时间复杂度：O(nlogk)
		n 是数据流元素个数，建堆操作耗时 O(k)，每次调整堆结构耗时 O(logk)
		总的时间复杂度是 O(nlogk)
	空间复杂度：O(k)
*/
type KthLargest struct {
	myHeap *MyHeap
	k int
}
func Constructor(k int, nums []int) KthLargest {
	n := len(nums)
	myHeap := &MyHeap{}
	heap.Init(myHeap)
	for i := 0; i < n; i ++ {
		heap.Push(myHeap, nums[i])
		// 把多余元素弹出
		if myHeap.Len() > k {
			heap.Pop(myHeap)
		}
	}
	return KthLargest{myHeap, k}
}
func (this *KthLargest) Add(val int) int {
	heap.Push(this.myHeap, val)
	if this.myHeap.Len() > this.k {
		heap.Pop(this.myHeap)
	}
	// 返回堆顶元素
	return (*this.myHeap)[0]
}

type MyHeap []int
// 实现 sort.Interface
func (h MyHeap) Len() int {
	return len(h)
}
func (h MyHeap) Less(i, j int) bool {
	return h[i] < h[j]
}
func (h MyHeap) Swap(i, j int) {
	h[i], h[j] = h[j], h[i]
}
// 实现 heap.Interface
func (this *MyHeap) Push(x interface{}) {
	*this = append(*this, x.(int))
}
func (this *MyHeap) Pop() interface{} {
	x := (*this)[this.Len() - 1]
	(*this) = (*this)[: this.Len() - 1]
	return x
}

/* 
	方法三：优先队列
	思路：
		使用一个大小为 k 的优先队列来存储前 k 大的元素，其中优先队列的队
		头为队列中最小的元素，也就是第 k 大的元素。
		在单次插入的操作中，我们首先将元素 val 加入到优先队列中。如果此时
		优先队列的大小大于 k，我们需要将优先队列的队头元素弹出，以保证优先
		队列的大小为 k。

		优先队列最常见的实现方法就是用小顶堆。此处采用继承 sort.IntSlice 
		的方式来实现 sort.Interface 接口，再实现 Push、Pop 方法，如此就
		相当于是实现了 heap.Interface 接口，从而可以进行小顶堆的操作。
	时间复杂度：O(nlog⁡k)
		初始化时间复杂度为：O(nlog⁡k) ，其中 n 为初始化时 nums 的长度；
		单次插入时间复杂度为：O(log⁡k)，插入之后需要对队列进行排序。
	空间复杂度：O(k)
		需要使用优先队列存储前 k 大的元素。
*/
type KthLargest struct {
	// 用继承 sort.IntSlice 的方式来实现 sort.Interface 接口
    sort.IntSlice
    k int
}
// 实现 heap.Interface 接口
func (kl *KthLargest) Push(v interface{}) {
    kl.IntSlice = append(kl.IntSlice, v.(int))
}
func (kl *KthLargest) Pop() interface{} {
    a := kl.IntSlice
    v := a[len(a)-1]
    kl.IntSlice = a[:len(a)-1]
    return v
}
func Constructor(k int, nums []int) KthLargest {
    kl := KthLargest{k: k}
    for _, val := range nums {
        kl.Add(val)
    }
    return kl
}
func (kl *KthLargest) Add(val int) int {
    heap.Push(kl, val)
    if kl.Len() > kl.k {
        heap.Pop(kl)
    }
    return kl.IntSlice[0]
}

/*
========================== 5、二叉搜索树的最近公共祖先 ==========================
给定一个二叉搜索树, 找到该树中两个指定节点的最近公共祖先。
百度百科中最近公共祖先的定义为：“对于有根树 T 的两个结点 p、q，最近公共祖
先表示为一个结点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（一个节点也
可以是它自己的祖先）。”
例如，给定如下二叉搜索树:  root = [6,2,8,0,4,7,9,null,null,3,5]
		 6
	   /  \
	  2    8
	 / \  / \
	0   4 7  9
       / \
	  3   5
	   
示例 1:
输入: root = [6,2,8,0,4,7,9,null,null,3,5], p = 2, q = 8
输出: 6 
解释: 节点 2 和节点 8 的最近公共祖先是 6。

示例 2:
输入: root = [6,2,8,0,4,7,9,null,null,3,5], p = 2, q = 4
输出: 2
解释: 节点 2 和节点 4 的最近公共祖先是 2, 因为根据定义最近公共祖先节点可
以为节点本身。

说明:
    所有节点的值都是唯一的。
    p、q 为不同节点且均存在于给定的二叉搜索树中。

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/introduction-to-data-structure-binary-search-tree/xpf523/
*/
/* 
	方法一：递归
	思路：
		我们递归遍历整棵二叉树，定义 fx​ 表示 x 节点的子树中是否包含 p 节点
		或 q 节点，如果包含为 true，否则为 false。那么符合条件的最近公共祖
		先 x 一定满足如下条件：
			(flson && frson) || ((x = p || x = q) && (flson || frson))
		其中 lson和 rson 分别代表 x 节点的左孩子和右孩子。初看可能会感觉条
		件判断有点复杂，我们来一条条看:
			1、(flson && frson)
				flson && frson 说明左子树和右子树均包含 p 节点或 q 节点，
				如果左子树包含的是 p 节点，那么右子树只能包含 q 节点，反之
				亦然，因为 p 节点和 q 节点都是不同且唯一的节点，因此如果满
				足这个判断条件即可说明 x 就是我们要找的最近公共祖先。
			2、((x = p || x = q) && (flson || frson))
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
	方法三：两次遍历
	思路：
		已知题目所给的树是二叉搜索树，那么我们便可以按二叉搜索树的性质去搜
		索目标节点：
			1、val = root.Val，找到目标节点
			2、val < root.Val，往左子树找，即 root = root.Left
			3、val > root.Val，往右子树找，即 root = root.Right
		首先我们先遍历一次二叉搜索树寻找 p 节点，在遍历过程中我们记录好走
		过的路径，接着再遍历二叉搜索树寻找 q 节点，在寻找 q 节点的过程中
		如果偏离了寻找 p 节点的路径，即出现了“分叉”，那么“分叉点”就是 p
		和 q 的最近公共祖先，返回该公共祖先节点。
	时间复杂度：O(n)
		n 是二叉搜索树的节点个数，此算法与 p、q 在树中的深度线性相关，
		而在最坏的情况下，树呈现链式结构，p 和 q 一个是树的唯一叶子结点，
		一个是该叶子结点的父节点，此时时间复杂度为 O(n)。
	空间复杂度：O(n)
		我们需要存储根节点到 p 和 q 的路径。和上面的分析方法相同，在最坏
		的情况下，路径的长度为 O(n)，因此需要 O(n) 的空间。
*/
func getPath(root, target *TreeNode)(path []*TreeNode) {
	node := root
    for node != target {
        path = append(path, node)
        if target.Val < node.Val {
            node = node.Left
        } else {
            node = node.Right
        }
    }
    path = append(path, node)
    return
}
func lowestCommonAncestor(root, p, q *TreeNode) *TreeNode {
	// 一次遍历获取搜索 p 时所走过的路径
	pathP := getPath(root, p)
	// 二次遍历获取搜索 q 时所走过的路径
	pathQ := getPath(root, q)
	// 寻找分叉点
	parent := pathP[0]
    for i := 0; i < len(pathP) && i < len(pathQ) && pathP[i] == pathQ[i]; i++ {
        parent = pathP[i]
    }
    return parent
}

/* 
	方法四：一次遍历
	思路：
		依旧利用二叉搜索树的性质来寻找分叉点，但只遍历一次。
		现在我们这样做，在二叉树中同时搜索 p、q：
			1、如果 p < root.Val && q < root.Val，往左子树找
				root = root.Left
			2、如果 p > root.Val && q > root.Val，往右子树找
				root = root.Right
			3、否则说明 p、q 在此产生分叉，返回当前节点。
				return root
	时间复杂度：O(n)
		n 是二叉搜索树的节点个数，此算法与 p、q 在树中的深度线性相关，
		而在最坏的情况下，树呈现链式结构，p 和 q 一个是树的唯一叶子结点，
		一个是该叶子结点的父节点，此时时间复杂度为 O(n)。
	空间复杂度：O(1)
		我们无需再记录搜索路径。
*/
func lowestCommonAncestor(root, p, q *TreeNode) *TreeNode {
	for root != nil {
		if p.Val < root.Val && q.Val < root.Val {
			root = root.Left
		} else if p.Val > root.Val && q.Val > root.Val {
			root = root.Right
		} else {
			return root
		}
	}
	return root
}

/* 
========================== 6、存在重复元素 III ==========================
给你一个整数数组 nums 和两个整数 k 和 t 。请你判断是否存在 两个不同下
标 i 和 j，使得 abs(nums[i] - nums[j]) <= t ，同时又满足 
abs(i - j) <= k 。
如果存在则返回 true，不存在返回 false。

示例 1：
输入：nums = [1,2,3,1], k = 3, t = 0
输出：true

示例 2：
输入：nums = [1,0,1,1], k = 1, t = 2
输出：true

示例 3：
输入：nums = [1,5,9,1,5,9], k = 2, t = 3
输出：false

提示：
    0 <= nums.length <= 2 * 10^4
    -231 <= nums[i] <= 231 - 1
    0 <= k <= 10^4
    0 <= t <= 2^31 - 1

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/introduction-to-data-structure-binary-search-tree/xpffam/
*/
/* 
	方法一：暴力
	思路：
		我们可以在限制条件下对数组进行遍历，找到符合条件的目标时返回。
	时间复杂度：O(n^2)
	空间复杂度：O(1)
*/
func containsNearbyAlmostDuplicate(nums []int, k int, t int) bool {
	n := len(nums)
	for i := 0; i < n; i ++ {
		for j := i + 1; j < n; j ++ {
			if abs(i, j) <= k && abs(nums[i], nums[j]) <= t {
				return true
			}
		}
	}
	return false
}
func abs(a, b int) int {
	if a > b {
		return a - b
	}
	return b - a
}

/* 
	方法二：滑动窗口 + 有序集合
	思路：
		对于序列中每一个元素 x 左侧的至多 k 个元素，如果这 k 个元素中存
		在一个元素落在区间 [x−t,x+t] 中，我们就找到了一对符合条件的元素。
		注意到对于两个相邻的元素，它们各自的左侧的 k 个元素中有 k−1 个是
		重合的。于是我们可以使用滑动窗口的思路，维护一个大小为 k 的滑动
		窗口，每次遍历到元素 x 时，滑动窗口中包含元素 x 前面的最多 k 个
		元素，我们检查窗口中是否存在元素落在区间 [x−t,x+t] 中即可。

		如果使用队列维护滑动窗口内的元素，由于元素是无序的，我们只能对于
		每个元素都遍历一次队列来检查是否有元素符合条件。如果数组的长度
		为 n，则使用队列的时间复杂度为 O(nk)，会超出时间限制。

		因此我们希望能够找到一个数据结构维护滑动窗口内的元素，该数据结构
		需要满足以下操作：
			支持添加和删除指定元素的操作，否则我们无法维护滑动窗口；
			内部元素有序，支持二分查找的操作，这样我们可以快速判断滑动窗
			口中是否存在元素满足条件，具体而言，对于元素 x，当我们希望判
			断滑动窗口中是否存在某个数 y 落在区间 [x−t,x+t] 中，只需要
			判断滑动窗口中所有大于等于 x−t 的元素中的最小元素是否小于等
			于 x+t 即可。

		我们可以使用有序集合来支持这些操作。
		实现方面，我们在有序集合中查找大于等于 x−t 的最小的元素 y，如果
		y 存在，且 y≤x+t，我们就找到了一对符合条件的元素。完成检查后，
		我们将 x 插入到有序集合中，如果有序集合中元素数量超过了 k，我们
		将有序集合中最早被插入的元素删除即可。
	注意：
		如果当前有序集合中存在相同元素，那么此时程序将直接返回 true。因
		此本题中的有序集合无需处理相同元素的情况。

		由此，本题中的有序集合我们可以选用二叉搜索树来实现。
*/
type TreeNode {
	Val int
	Left, Right *TreeNode
}
// 往二叉搜索树中插入节点
func insert(root *TreeNode, val int) (*TreeNode, bool) {
	node := &TreeNode{Val : val}
	if root == nil {
		root = node
		return root, true
	}
	parent, cur := root, root
	for cur != nil {
		// 元素已存在
		if val == cur.Val {
			return root, false
		}
		parent = cur
		if val < cur.Val {
			cur = cur.Left
		} else {
			cur = cur.Right
		}
	}
	if val < cur.Val {
		parent.Left = node
	} else {
		parent.Right = node
	}
	return root, true
}

// 删除二叉搜索树的节点
func deleteNode(root *TreeNode, val int) *TreeNode {
	if root == nil {
		return nil
	}
	// 找到要删除的节点
	if val == root.Val {
		// 没有节点或只有右节点
		if root.Left == nil {
			return root.Right
		}
		// 只有左节点
		if root.Right == nil {
			return root.Left
		}
		// 同时有左右节点
		// 找到被删除节点的中序后继节点（右子树中最左的节点）
		cur := root.Right
		for cur.Left != nil {
			cur = cur.Left
		}
		// 把被删除节点的左子树挂载在它的中序后继节点的左子树下
		cur.Left = root.Left
		// 返回被删除节点的右子树，相当于把 root 节点删除了
		return root.Right
	}
	if val < root.Val {
		root.Left = deleteNode(root.Left, val)
	} else {
		root.Right = deleteNode(root.Right, val)
	}
	return root
}
func containsNearbyAlmostDuplicate(nums []int, k int, t int) bool {
	// 滑动窗口元素
	var window *TreeNode
	L, R := 0, 0
	for i, v := range nums {
		// 窗口扩张
		R ++
		insert(window, nums[i])
		// 窗口收缩
		for R - L + 1 > k {
			deleteNode(window, nums[L])
			L ++
		}
		// 查找符合条件的元素
		// 查找大于等于 x−t 的最小的元素 y，如果 y 存在，且 y≤x+t，返回 true
		cur := window
		for cur != nil {
			if nums[i] - t <= cur.Val {
				
			}
		}
	}

	return false
}