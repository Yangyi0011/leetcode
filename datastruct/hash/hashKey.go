package hash

/* 
	哈希表 key 的设计
		设计关键是在原始信息和哈希映射使用的实际键之间建立映射关系。
		设计键时，需要保证：
			1. 属于同一组的所有值都将映射到同一组中。
			2. 需要分成不同组的值不会映射到同一组。
*/
/* 
========================== 1、字母异位词分组 ==========================
给定一个字符串数组，将字母异位词组合在一起。字母异位词指字母相同，但排列
不同的字符串。

示例:
输入: ["eat", "tea", "tan", "ate", "nat", "bat"]
输出:
[
  ["ate","eat","tea"],
  ["nat","tan"],
  ["bat"]
]

说明：
    所有输入均为小写字母。
    不考虑答案输出的顺序。

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/group-anagrams
*/
/* 
	方法一：哈希表法
	思路：
		本题的关键在于如何给字母异位词分组，对于分组问题，我们很容易就能
		想到用哈希映射，但难点在于这个映射的 key 该如何设计。
		
		由于互为字母异位词的两个字符串包含的字母相同，因此对两个字符串分
		别进行排序之后得到的字符串一定是相同的，故可以将排序之后的字符串
		作为哈希表的键。然后在遍历字符数组进行字符串分组的过程中，
		我们把排序结果一致的字符串分配到同一组中。如：
			"ate"、"eat"、"tea" 它们排序后的 key 是 "aet"，所以它们都
			会被分到同一组
	时间复杂度：O(n*k*logk)
		n 是字符数组的元素个数，k 是数组中字符串的平均长度.
		我们遍历分组字符串耗时 O(n)，而对每一个字符串进行排序耗
		时 O(k*logk)，所以总耗时为 O(n*k*logk)
	空间复杂度：O(n*k)
		n 是字符数组的元素个数，k 是数组中字符串的平均长度.
		我们需要用哈希表存储全部字符串。
*/
type Bytes []byte
func (this Bytes) Len() int {
	return len(this)
}
func (this Bytes) Less(i, j int) bool {
	return this[i] < this[j]
}
func (this Bytes) Swap(i, j int) {
	this[i], this[j] = this[j], this[i]
}
func groupAnagrams(strs []string) [][]string {
	hashMap := make(map[string][]string, 0)
	for _, v := range strs {
		arr := []byte(v)
		sort.Sort(Bytes(arr))
		key := string(arr)
		if _, ok := hashMap[key]; !ok {
			hashMap[key] = make([]string, 0)
		}
		hashMap[key] = append(hashMap[key], v)
	}
	res := make([][]string, 0)
	for _, v := range hashMap {
		res = append(res, v)
	}
	return res
}

/* 
	方法二：计数
	思路：
		由于互为字母异位词的两个字符串包含的字母相同，因此两个字符串中
		的相同字母出现的次数一定是相同的，故可以将每个字母出现的次数使
		用字符串表示，作为哈希表的键。
		由于字符串只包含小写字母，因此对于每个字符串，可以使用长度为 26
		的数组记录每个字母出现的次数。
	时间复杂度：O(n(k+∣Σ∣))
		其中 n 是 strs 中的字符串的数量，k 是 strs 中的字符串的的最大
		长度，Σ 是字符集，在本题中字符集为所有小写字母，∣Σ∣=26|。
		需要遍历 n 个字符串，对于每个字符串，需要 O(k) 的时间计算每个
		字母出现的次数，O(∣Σ∣) 的时间生成哈希表的键，以及 O(1) 的时间更
		新哈希表，因此总时间复杂度是 O(n(k+∣Σ∣))。
	空间复杂度：O(n(k+∣Σ∣))
		其中 n 是 strs 中的字符串的数量，k 是 strs 中的字符串的最大长
		度，Σ 是字符集，在本题中字符集为所有小写字母，∣Σ∣=26。需要用哈
		希表存储全部字符串，而记录每个字符串中每个字母出现次数的数组需
		要的空间为 O(∣Σ∣)，在渐进意义下小于 O(n(k+∣Σ∣))，可以忽略不计。
*/
func groupAnagrams(strs []string) [][]string {
	// 注意，切片不能作为 key
	hashMap := make(map[[26]int][]string, 0)
	for _, v := range strs {
		var key [26]int = [26]int{}
		// 记录字符串每个字符出现的次数
		for _, c := range []byte(v) {
			key[c - 'a'] ++
		}
		if _, ok := hashMap[key]; !ok {
			hashMap[key] = make([]string, 0)
		}
		hashMap[key] = append(hashMap[key], v)
	}
	res := make([][]string, 0)
	for _, v := range hashMap {
		res = append(res, v)
	}
	return res
}

/* 
========================== 2、有效的数独 ==========================
判断一个 9x9 的数独是否有效。只需要根据以下规则，验证已经填入的数字是
否有效即可。
    数字 1-9 在每一行只能出现一次。
    数字 1-9 在每一列只能出现一次。
    数字 1-9 在每一个以粗实线分隔的 3x3 宫内只能出现一次。
上图是一个部分填充的有效的数独。
数独部分空格内已填入了数字，空白格用 '.' 表示。

示例 1:
输入:
[
  ["5","3",".",".","7",".",".",".","."],
  ["6",".",".","1","9","5",".",".","."],
  [".","9","8",".",".",".",".","6","."],
  ["8",".",".",".","6",".",".",".","3"],
  ["4",".",".","8",".","3",".",".","1"],
  ["7",".",".",".","2",".",".",".","6"],
  [".","6",".",".",".",".","2","8","."],
  [".",".",".","4","1","9",".",".","5"],
  [".",".",".",".","8",".",".","7","9"]
]
输出: true

示例 2:
输入:
[
  ["8","3",".",".","7",".",".",".","."],
  ["6",".",".","1","9","5",".",".","."],
  [".","9","8",".",".",".",".","6","."],
  ["8",".",".",".","6",".",".",".","3"],
  ["4",".",".","8",".","3",".",".","1"],
  ["7",".",".",".","2",".",".",".","6"],
  [".","6",".",".",".",".","2","8","."],
  [".",".",".","4","1","9",".",".","5"],
  [".",".",".",".","8",".",".","7","9"]
]
输出: false
解释: 除了第一行的第一个数字从 5 改为 8 以外，空格内其他数字均与 示例1 相同。
     但由于位于左上角的 3x3 宫内有两个 8 存在, 因此这个数独是无效的。

说明:
    一个有效的数独（部分已被填充）不一定是可解的。
    只需要根据以上规则，验证已经填入的数字是否有效即可。
    给定数独序列只包含数字 1-9 和字符 '.' 。
    给定数独永远是 9x9 形式的。

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/valid-sudoku
*/
/* 
	方法一：一次遍历
	思路：
		确定是不是数独有以下3个关键：
			1、行上的数字不能重复
			2、列上的数字不能重复
			3、3x3的子数独上的数字不能重复
		如果不符合上面任意条件，则返回 false。难点在于子数独的不重复确定。
		如何枚举子数独？
			可以使用 box_index = (row/3)*3 + column / 3， / 是整数
			除法。
		时间复杂度：O(1)
			因为我们只对 81 个单元格进行了一次迭代。
		空间复杂度：O(1)
*/
func isValidSudoku(board [][]byte) bool {
	// 因为行、列、子数独都只有 9 个元素，所以我们不用 hash 表
	// 来记录重复，而是用数组对应的下标来记录这个值是否已经出现过即可
	// 第一维度：当前行、列、子数独，第二维度：当前数字是否已存在
	row := make([][]bool, 9)
	column := make([][]bool, 9)
	box := make([][]bool, 9)
	for i := 0; i < 9; i ++ {
		row[i] = make([]bool, 9)
		column[i] = make([]bool, 9)
		box[i] = make([]bool, 9)
	}
	for i := 0; i < len(board); i ++ {
		for j := 0; j < len(board[i]); j ++ {
			if board[i][j] == '.' {
				continue
			}
			// 获取该字符对应的数字，因为数独的数字是 1-9，而下标的
			// 数字是 0-8，所以需要 - '1'
			num := board[i][j] - '1'
			// 计算子数独的下标
			boxIndex := (i / 3) * 3 + j / 3
			// 如果在该数在当前行、列、子数独中已经出现过，则返回 false
			if row[i][num] || column[j][num] || box[boxIndex][num] {
				return false
			}
			row[i][num] = true
			column[j][num] = true
			box[boxIndex][num] = true
		}
	}
	return true
}

/* 
========================== 3、寻找重复的子树 ==========================
给定一棵二叉树，返回所有重复的子树。对于同一类的重复子树，你只需要返回其
中任意一棵的根结点即可。
两棵树重复是指它们具有相同的结构以及相同的结点值。

示例 1：
        1
       / \
      2   3
     /   / \
    4   2   4
       /
      4

下面是两个重复的子树：

      2
     /
    4
和
    4
因此，你需要以列表的形式返回上述重复子树的根结点。

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/hash-table/xxm0i6/
*/
/**
 * Definition for a binary tree node.
 * type TreeNode struct {
 *     Val int
 *     Left *TreeNode
 *     Right *TreeNode
 * }
 */
 /* 
	方法一：DFS
	思路：
		序列化二叉树。
			 1
			/ \
		   2   3
			  / \
			 4   5
		例如上面这棵树序列化结果为 1,2,#,#,3,4,#,#,5,#,#。每棵不同子树
		的序列化结果都是唯一的。
		使用深度优先搜索，其中递归函数返回当前子树的序列化结果。把每个节
		点开始的子树序列化结果保存在哈希表中，然后判断是否存在重复的子树。
	时间复杂度：O(n^2))
		其中 n 是二叉树上节点的数量。遍历所有节点，在每个节点处序列化需
		要时间 O(n)。
	空间复杂度：O(n^2)
		其中 n 是二叉树上节点的数量，主要是哈希表的大小。
 */
func findDuplicateSubtrees(root *TreeNode) []*TreeNode {
	hash := make(map[string]int, 0)
	result := make([]*TreeNode, 0)
	var DFS func(root *TreeNode) string
	DFS = func(root *TreeNode) string {
		// nil & leaf
		if root == nil {
			return "#"
		}
		// Divide
		left := DFS(root.Left)
		right := DFS(root.Right)
		// Conquer
		serial := fmt.Sprintf("%d,%s,%s", root.Val, left, right)
		hash[serial] ++
		if hash[serial] == 2 {
			result = append(result, root)
		}
		return serial
	}
	DFS(root)
	return result
}

/* 
	方法二：唯一标识符
	思路：
		假设每棵子树都有一个唯一标识符：只有当两个子树的 id 相同时，
		认为这两个子树是相同的。
		一个节点 node 的左孩子 id 为 x，右孩子 id 为 y，那么该节点
		的 id 为 (node.val, x, y)

		如果三元组 (node.val, x, y) 第一次出现，则创建一个这样的三元
		组记录该子树。如果已经出现过，则直接使用该子树对应的 id。
	时间复杂度：O(N)
		其中 N 二叉树上节点的数量，每个节点都需要访问一次。
	空间复杂度：O(N)
		每棵子树的存储空间都为 O(1)。
*/
func findDuplicateSubtrees(root *TreeNode) []*TreeNode {
	// 记录子树序列及其唯一id，key：序列，value：id
	trees := make(map[string]int, 0)
	// 记录唯一id
	count := make(map[int]int, 0)
	result := make([]*TreeNode, 0)
	// 自增 id
	uuid := 1
	var DFS func(root *TreeNode) int
	DFS = func(root *TreeNode) int {
		if root == nil {
			return 0
		}
		left := DFS(root.Left)
		right := DFS(root.Right)
		serial := fmt.Sprintf("%d,%s,%s", root.Val, left, right)
		if _, ok := trees[serial]; !ok {
			// 如果是新的子树，则 uuid ++
			uuid ++
			// 给对应的子树分配 id
			trees[serial] = uuid
		}
		// 记录对应 id 的子树出现的次数
		count[trees[serial]]++
		if count[trees[serial]] == 2 {
			result = append(result, root)
		}
		return trees[serial]
	}
	DFS(root)
	return result
}