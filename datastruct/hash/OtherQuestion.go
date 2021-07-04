package hash

/* 
	关于哈希表的常见问题
*/

/* 
========================== 1、宝石与石头 =========================
给定字符串J 代表石头中宝石的类型，和字符串 S代表你拥有的石头。 S 中每个
字符代表了一种你拥有的石头的类型，你想知道你拥有的石头中有多少是宝石。
J 中的字母不重复，J 和 S中的所有字符都是字母。字母区分大小写，因此"a"和
"A"是不同类型的石头。

示例 1:
输入: J = "aA", S = "ABb"
输出: 3

示例 2:
输入: J = "z", S = "ZZ"
输出: 0

注意:
    S 和 J 最多含有50个字母。
     J 中的字符不重复。

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/hash-table/xx2a0c/
*/
/* 
	方法一：哈希表法
	思路：
		用哈希表记录 S 中宝石的个数，以字符串 J 的字符作为 key，J的字符
		在 S 中出现的次数作为 value，遍历字符串 S 即可得到每一种宝石的数
		量，最后再遍历哈希表把所有宝石的数量相加即得到结果。
	时间复杂度：O(m+n)
		m、n 分别是字符串 J 和 S 的长度，我们需要先遍历 J 完成哈希表 key
		的处理，再遍历 S 完成 哈希表 value 的计算。
	空间复杂度：O(m)
		m 是字符串 J 的长度。
*/
func numJewelsInStones(jewels string, stones string) int {
	m := len(jewels)
	if m == 0 {
		return 0
	}
	hash := make(map[byte]int, m)
	for i := 0; i < m; i ++ {
		hash[jewels[i]] = 0
	}
	for i := 0; i < len(stones); i ++ {
		if _, ok := hash[stones[i]]; ok {
			hash[stones[i]] ++
		}
	}
	ans := 0
	for _, v := range hash {
		ans += v
	}
	return ans
}

/* 
	方法二：哈希表法【优化】
	思路：
		经过分析题目可知，我们根本不需要记录每一种宝石的数量，只需计算宝石
		的总数量即可，如此我们只需要记录每一个字符是不是宝石即可。
	时间复杂度：O(m+n)
		m、n 分别是字符串 J 和 S 的长度，我们需要先遍历 J 确定宝石字符，
		再遍历 S 完成结果的计算。
	空间复杂度：O(m)
		m 是字符串 J 的长度。
*/
func numJewelsInStones(jewels string, stones string) int {
	m := len(jewels)
	hash := make(map[byte]bool, m)
	for i := 0; i < m; i ++ {
		hash[jewels[i]] = true
	}
	ans := 0
	for i := 0; i < len(stones); i ++ {
		if hash[stones[i]] {
			ans ++
		}
	}
	return ans
}

/* 
========================== 2、无重复字符的最长子串 =========================
给定一个字符串，请你找出其中不含有重复字符的 最长子串 的长度。

示例 1:
输入: s = "abcabcbb"
输出: 3 
解释: 因为无重复字符的最长子串是 "abc"，所以其长度为 3。

示例 2:
输入: s = "Bbb"
输出: 1
解释: 因为无重复字符的最长子串是 "b"，所以其长度为 1。

示例 3:
输入: s = "pwwkew"
输出: 3
解释: 因为无重复字符的最长子串是 "wke"，所以其长度为 3。
     请注意，你的答案必须是 子串 的长度，"pwke" 是一个子序列，不是子串。

示例 4:
输入: s = ""
输出: 0

提示：
    0 <= s.length <= 5 * 10^4
    s 由英文字母、数字、符号和空格组成

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/hash-table/xxnrdi/
*/
/* 
	方法一：滑动窗口
	思路：
		定义 L、R 两个指针来表示窗口的左右边界，即无重复最长子串的左右
		边界，然后用一个哈希表来记录窗口中的元素出现的次数。遍历字符串，
		如果当前字符在窗口中没有出现过，则标记当前字符为已出现并扩展
		右边界，否则收缩左边界并移除收缩过程中的出现标记，然后计算当前
		最长子串长度。
	时间复杂度：O(n)
		其中 n 是字符串的长度。左指针和右指针分别会遍历整个字符串一次。
	空间复杂度：O(∣Σ∣)
		其中 |Σ| 表示字符集（即字符串中可以出现的字符），∣Σ∣ 表示字符集
		的大小。在本题中没有明确说明字符集，因此可以默认为所有 ASCII 码
		在 [0,128) 内的字符，即 ∣Σ∣=128。我们需要用到哈希集合来存储出
		现过的字符，而字符最多有 ∣Σ∣| 个，因此空间复杂度为 O(∣Σ∣)。
*/
func lengthOfLongestSubstring(s string) int {
	n := len(s)
	if n == 0 {
		return 0
	}
	hash := make(map[byte]bool)
	L, R := 0, 0
	ans := 0
	for R < n {
		for hash[s[R]] {
			// 计算无重复最长子串
			ans = max(ans, R - L)
			// 收缩左边界
			delete(hash, s[L])
			L ++
		} 
        // 扩展右边界
		hash[s[R]] = true
        R ++
	}
	// 计算最后一段
	return max(ans, R - L)
}
func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
/* 
	方法二：滑动窗口【模板】
	思路：
		使用滑动窗口来查找【无重复字符的最长子串】，确保窗口中的字符不重
		复出现，如果有字符重复出现，则收缩窗口。
	时间复杂度：O(n)
		n 表示字符串的长度
	空间复杂度：O(k)
		k 表示英文字母、数字、符号、空格的字符数量（每个字符只算一次，
		不重复计算）
*/
func lengthOfLongestSubstring(s string) int {
	n := len(s)
	if n == 0 {
		return 0
	}
	window := make(map[byte]int, 0)
	L, R, ans := 0, 0, 0
	for R < n {
		c := s[R]
		R ++ 
		window[c] ++
		// 字符频率大于1，则收缩窗口
		for window[c] > 1 {
			// c 变量不能在此处重用
			ch := s[L]
			L ++
			window[ch] --
		}
		ans = max(ans, R - L)
	}
	return ans
}

/* 
========================== 3、四数相加 II =========================
给定四个包含整数的数组列表 A , B , C , D ,计算有多少个元组 (i, j, k, l) ，
使得 A[i] + B[j] + C[k] + D[l] = 0。
为了使问题简单化，所有的 A, B, C, D 具有相同的长度 N，且 0 ≤ N ≤ 500 。
所有整数的范围在 -2^28 到 2^28 - 1 之间，最终结果不会超过 2^31 - 1 。

例如:
输入:
A = [ 1, 2]
B = [-2,-1]
C = [-1, 2]
D = [ 0, 2]

输出:
2

解释:
两个元组如下:
1. (0, 0, 0, 1) -> A[0] + B[0] + C[0] + D[1] = 1 + (-2) + (-1) + 2 = 0
2. (1, 1, 0, 0) -> A[1] + B[1] + C[0] + D[0] = 2 + (-1) + (-1) + 0 = 0

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/hash-table/xxwhng/
*/
/* 
	发方法一：暴力【超时】
	思路：
		四重循环遍历，对四个数组的每一个元素都进行计算。
	时间复杂度：O(n^4)
	空间复杂度：O(1)
*/
func fourSumCount(nums1 []int, nums2 []int, nums3 []int, nums4 []int) int {
	n := len(nums1)
	ans := 0
	for i := 0; i < n; i ++ {
		for j := 0; j < n; j ++ {
			for k := 0; k < n; k ++ {
				for l := 0; l < n; l ++ {
					if nums1[i] + nums2[j] + nums3[k] + nums4[l] == 0 {
						ans ++
					}
				}
			}
		}	
	}
	return ans
}

/* 
	方法二：分组+哈希表
	思路：
		我们可以将四个数组分成两部分，A 和 B 为一组，C 和 D 为另外一组。
		对于 A 和 B，我们使用二重循环对它们进行遍历，得到所有 A[i]+B[j]
		的值并存入哈希映射中。对于哈希映射中的每个键值对，每个键表示一种 
		A[i]+B[j]，对应的值为 A[i]+B[j] 出现的次数。

		对于 C 和 D，我们同样使用二重循环对它们进行遍历。当遍历到 C[k]+D[l]
		时，如果 −(C[k]+D[l]) 出现在哈希映射中，那么将 −(C[k]+D[l]) 
		对应的值累加进答案中。
		最终即可得到满足 A[i]+B[j]+C[k]+D[l]=0 的四元组数目。
	时间复杂度：O(n^2)
		我们使用了两次二重循环，时间复杂度均为 O(n2)。在循环中对哈希映
		射进行的修改以及查询操作的期望时间复杂度均为 O(1)，因此总时间
		复杂度为 O(n^2)。
	空间复杂度：O(n^2)
		即为哈希映射需要使用的空间。在最坏的情况下，A[i]+B[j] 的值均不
		相同，因此值的个数为 n^2，也就需要 O(n2) 的空间。
*/
func fourSumCount(nums1 []int, nums2 []int, nums3 []int, nums4 []int) int {
	hash := make(map[int]int)
	for _, A := range nums1 {
		for _, B := range nums2 {
			// 记录相同 key 出现的次数
			hash[A + B] ++
		}
	}
	ans := 0
	for _, C := range nums3 {
		for _, D := range nums4 {
			if v, ok := hash[-(C + D)]; ok {
				// 需要把第一组中相同 key 出现的次数加上
				ans += v
			}
		}
	}
	return ans
}

/* 
========================== 4、前 K 个高频元素 =========================
给你一个整数数组 nums 和一个整数 k ，请你返回其中出现频率前 k 高的元素。
你可以按 任意顺序 返回答案。

示例 1:
输入: nums = [1,1,1,2,2,3], k = 2
输出: [1,2]

示例 2:
输入: nums = [1], k = 1
输出: [1]

提示：
    1 <= nums.length <= 10^5
    k 的取值范围是 [1, 数组中不相同的元素的个数]
    题目数据保证答案唯一，换句话说，数组中前 k 个高频元素的集合是唯一的

进阶：你所设计算法的时间复杂度 必须 优于 O(nlog n) ，其中 n 是数组大小。

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/hash-table/xxwb2v/
*/
/* 
	方法一：哈希表法 + 排序
	思路：
		先用哈希表存储每一个数出现的次数，然后按出现次数进行排序，取出现
		次数最大的前 k 个。
	时间复杂度：O(n logn)
		n 是数组的长度，我们需要先遍历数组记录每个数的出现次数，然后再对
		哈希表按出现次数进行排序。
	空间复杂度：O(n)
		n 是数组的长度，我们需要用哈希表记录不同数字的出现次数，最坏情
		况下数组的每个数都不同，此时我们需要记录所有数字。
*/
type entity struct {
	key, value int
}
type entities []*entity
func (es entities) Len() int {
	return len(es)
}
func (es entities) Less(i, j int) bool {
	return es[i].value > es[j].value
}
func (es entities) Swap(i, j int) {
	es[i], es[j] = es[j], es[i]
}
func topKFrequent(nums []int, k int) []int {
	hash := make(map[int]int)
	for _, v := range nums {
		hash[v] ++
	}
	// 把 hash 的 k-v 转为结构体，对结构体数组进行排序
	es := make(entities, 0)
	for k, v := range hash {
		e := &entity{k, v}
		es = append(es, e)
	}
	sort.Sort(es)
	ans := make([]int, k)
	for i := 0; i < k; i ++ {
		ans[i] = es[i].key
	}
	return ans
}

/* 
	方法二：小根堆【手写小根堆】
	思路：
		首先遍历整个数组，并使用哈希表记录每个数字出现的次数，并形成一个
		「出现次数数组」hash。找出原数组的前 k 个高频元素，就相当于找出
		「出现次数数组」hash 的前 k 大的值。
		使用「出现次数数组」hash 的前 k 个元素构建长度为 k 的小根堆，接
		着遍历 hash 的后 n-k 个元素，对比每一个元素 hash[i] 与小根堆的
		堆顶元素，如果 hash[i] 大于堆顶元素，则把堆顶元素替换为 hash[i]，
		接着调整小根堆，重复上述操作，直到 hash 遍历完成。
	时间复杂度：O(n*logk)
	空间复杂度：O(k)
*/
func topKFrequent(nums []int, k int) []int {
	hash := make(map[int]int, 0)
	for _, v := range nums {
		hash[v] ++
	}
	heap := make([][2]int, 0)
	// 标记堆是否初始化
	init := false
	for key, value := range hash {
		if len(heap) < k {
			heap = append(heap, [2]int{key, value})
			continue
		}
		// 初始化堆
		if !init {
			buildHeap(heap)
			init = true
		}
		if value > heap[0][1] {
			// 替换堆顶元素和调整堆结构
			heap[0] = [2]int{key, value}
			heapFix(heap, 0, k)
		}
	}
	res := make([]int, k)
	for i := 0; i < k; i ++ {
        // 升序输出
		res[k - i - 1] = heap[i][0]
	}
	return res
}

// 构建小顶堆
func buildHeap(data [][2]int) {
	n := len(data)
	if n == 0 {
		return
	}
    // 从最后一个非叶子节点开始处理，一直处理到根节点
	lastFatherIndex := n/2 - 1
	for i := lastFatherIndex; i >= 0; i -- {
		heapFix(data, i, n)
	}
}

// 调整堆结构
// data 堆数据
// fatherIndex 从哪一个父节点开始调整
// n 需要调整的堆元素个数
func heapFix(data [][2]int, fatherIndex int, n int) {
	// 2 * fatherIndex + 1 从当前父节点的第一个子节点开始处理
    // i = 2*i+1 处理下一个父节点
	for i := 2 * fatherIndex + 1; i < n; i = 2*i + 1 {
		// 找到子节点中的最小值
		if i < n - 1 && data[i][1] > data[i + 1][1] {
			i ++
		}
		// 如果父节点是最小值，则不用处理直接跳出循环
		if data[fatherIndex][1] < data[i][1] {
			break
		}
		// 否则把最小子节点上浮到父节点的位置
		data[fatherIndex], data[i] = data[i], data[fatherIndex]
		// 【重要】父节点下标下沉
		fatherIndex = i
	}
}

/* 
	方法三：小根堆【使用接口】
	时间复杂度：O(n*logn)
	空间复杂度：O(k)
*/
func topKFrequent(nums []int, k int) []int {
	hash := make(map[int]int, 0)
	for _, v := range nums {
		hash[v] ++
	}
	// 初始化堆
	myHeap := &MyHeap{}
	heap.Init(myHeap)
	for key, value := range hash {
		// 注意，都是用 heap 包的函数来调用的
		heap.Push(myHeap, [2]int{key, value})
		// 如果堆内元素大于 k，则弹出堆顶元素
		if myHeap.Len() > k {
			heap.Pop(myHeap)
		}
	}
	res := make([]int, k)
	for i := 0; i < k; i ++ {
		res[k - i - 1] = (*myHeap)[i][0]
	}
	return res
}

// 自定义堆结构，继承 heap 包的 Interface 接口
type MyHeap [][2]int
// 继承自 sort.Interface
func (h MyHeap) Len() int {
	return len(h)
}
func (h MyHeap) Less(i, j int) bool {
	return h[i][1] < h[j][1]
}
func (h MyHeap) Swap(i, j int) {
	h[i], h[j] = h[j], h[i]
}
// 继承 heap.Interface
func (h *MyHeap) Push(x interface{}) {
	*h = append(*h, x.([2]int))
}
func (h *MyHeap) Pop() interface{} {
	x := (*h)[h.Len() - 1]
	*h = (*h)[: h.Len() - 1]
	return x
}

/* 
	方法四：基于快速排序
	思路：
		根据快排的性质：每一趟快排都将大于基值 K 的数放在 K 的右边，小于
		K 的数放在 K 的左边，由此每次快速排序中的划分过程定能找到一个全
		部大于左边元素的一个值 nums[i]。
		
		如果 len(nums)-1-i = K，那么这个值和它右边的所有元素就是 topK；
		如果 len(nums)-1-i > K，那么对右边的元素继续划分排序；
		如果 len(nums)-1-i < K，那么对左边的元素继续划分排序。
	时间复杂度：O(n^2)
		其中 n 是数组长度。
		设处理长度为 n 的数组的时间复杂度为 f(n)。由于处理的过程包括一次
		遍历和一次子分支的递归，最好情况下，有 f(N)=O(N)+f(N/2)，根据
		主定理，能够得到 f(N)=O(N)。
		最坏情况下，每次取的中枢数组的元素都位于数组的两端，时间复杂度退
		化为 O(N^2)。但由于我们在每次递归的开始会先随机选取中枢元素，故
		出现最坏情况的概率很低。
		平均情况下，时间复杂度为 O(n)。
	空间复杂度：O(n)
*/
func topKFrequent(nums []int, k int) []int {
	hash := map[int]int{}
    for _, num := range nums {
        hash[num]++
    }
    data := [][2]int{}
    for key, value := range hash {
        data = append(data, [2]int{key, value})
    }
	// 完成交换后，topK 就是 data[index:]
	index := quickSort(data, 0, len(data) - 1, k)
	res := make([]int, 0)
	for i := index; i < len(data); i ++ {
		res = append(res, data[i][0])
	}
    return res
}
// 通过快速排序原理来寻找 topK
// 返回第 k 大的数所在的下标
// data 数据
// start、end 每一轮快排的起始和终止下标
// k topK
func quickSort(data [][2]int, start, end, k int) int {
	// 选择随机基值的下标
	rand.Seed(time.Now().UnixNano())
    picked := rand.Int() % (end - start + 1) + start;
	data[picked], data[start] = data[start], data[picked]
	
	// 记录基值
	pivot := data[start][1]
	left, right := start, end
	// 一轮快排
	for (left < right) {
		// 先从右往左找小于基值的数，跳过大于等于基值的数
		for (left < right && data[right][1] >= pivot) {
			right --
		}
		// 找到了小于基值的数，把它交换到基值的前面
		if (data[right][1] < pivot) {
			data[left], data[right] = data[right], data[left]
			// 越过刚刚交换过来的元素，从左边开始找
			left ++
		}
		// 从左往右找大于基值的数，跳过小于等于基值的数
		for (left < right && data[left][1] <= pivot) {
			left ++
		}
		// 找到了大于基值的数，把它交换到基值的后面
		if (data[left][1] > pivot) {
			data[left], data[right] = data[right], data[left]
			// 越过刚刚交换过来的元素，从右边开始找
			right --
		}
	}
	// 获取基值下标往后的元素个数 rank
	rank := len(data) - left
	// 如果 rank 刚好是 k 个，说明找到了 topK 开始的下标
	if rank == k {
		return left
	}
	// rank < k，需要往左找
	if rank < k {
		return quickSort(data, start, left - 1, k)
	}
	// rank > k 往右找
	return quickSort(data, right + 1, end, k)
}

/* 
========================== 5、常数时间插入、删除和获取随机元素 =========================
设计一个支持在平均 时间复杂度 O(1) 下，执行以下操作的数据结构。
    insert(val)：当元素 val 不存在时，向集合中插入该项。
    remove(val)：元素 val 存在时，从集合中移除该项。
    getRandom：随机返回现有集合中的一项。每个元素应该有相同的概率被返回。

示例 :
// 初始化一个空的集合。
RandomizedSet randomSet = new RandomizedSet();

// 向集合中插入 1 。返回 true 表示 1 被成功地插入。
randomSet.insert(1);

// 返回 false ，表示集合中不存在 2 。
randomSet.remove(2);

// 向集合中插入 2 。返回 true 。集合现在包含 [1,2] 。
randomSet.insert(2);

// getRandom 应随机返回 1 或 2 。
randomSet.getRandom();

// 从集合中移除 1 ，返回 true 。集合现在包含 [2] 。
randomSet.remove(1);

// 2 已在集合中，所以返回 false 。
randomSet.insert(2);

// 由于 2 是集合中唯一的数字，getRandom 总是返回 2 。
randomSet.getRandom();

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/insert-delete-getrandom-o1
*/


/**
* Your RandomizedSet object will be instantiated and called as such:
* obj := Constructor();
* param_1 := obj.Insert(val);
* param_2 := obj.Remove(val);
* param_3 := obj.GetRandom();
*/

/* 
	方法一：动态数组 + 哈希表
	思路：
		直观感受是使用 哈希表，因为哈希表的 Insert 和 Remove 是 O(1)，
		但使用哈希表来处理 GetRandom 明显不可能，处理 GetRandom 需
		要先把 哈希表 转为 slice，再通过获取随机下标的方式来获取下标对
		应的值，但这样做 GetRandom 的时间复杂度就是 O(n)，不符合要求。

		我们可以考虑用 slice 来处理，slice 的 Remove 和 GetRandom 是
		O(1)，但是 Remove 却需要 O(n)，对此我们可以这样做，我们每次只
		删除最后一个元素，即找到要删除元素的下标，把它与最后一个元素交换，
		然后删除最后一个元素，为了能快速找到要删除元素的下标，我们还需要
		额外使用一个哈希表来记录 Val 和它在 slice 中的下标，以便把 
		Remove 操作的复杂度降低至 O(1)。
	时间复杂度：O(1)
		我们的所有操作都是 O(1)
	空间复杂度：O(n)
		我们需要用一个 slice 和 一个 hash 来做辅助处理。
*/
type RandomizedSet struct {
	slice []int
	hash map[int]int
}

/** Initialize your data structure here. */
func Constructor() RandomizedSet {
	return RandomizedSet{[]int{}, make(map[int]int, 0)}
}

/** Inserts a value to the set. Returns true if the set did not already contain the specified element. */
func (this *RandomizedSet) Insert(val int) bool {
	if _, ok := this.hash[val]; ok {
		return false
	}
	this.slice = append(this.slice, val)
	this.hash[val] = len(this.slice) - 1
	return true
}

/** Removes a value from the set. Returns true if the set contained the specified element. */
func (this *RandomizedSet) Remove(val int) bool {
	index, ok := this.hash[val]
	if !ok {
		return false
	}
	// 直接把最后一个元素的值覆盖到被删除元素的位置，再删除最后一个元素
	// 覆盖的效率比交换要高
	n := len(this.slice) - 1
	this.slice[index] = this.slice[n]
	// 更新 val-index 映射关系
	this.hash[this.slice[index]] = index
	// 删除最后元素
	this.slice = this.slice[:n]
	delete(this.hash, val)
	return true
}

/** Get a random element from the set. */
func (this *RandomizedSet) GetRandom() int {
	n := len(this.slice)
	//rand.Intn参数小于等于0导致panic
	if n == 0 {
		return -1
	}
	return this.slice[rand.Intn(n)]
}
