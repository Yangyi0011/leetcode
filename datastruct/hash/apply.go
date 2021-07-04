package hash

/* 
	hash 应用
*/
/* 
========================== 1、存在重复元素 ==========================
给定一个整数数组，判断是否存在重复元素。
如果存在一值在数组中出现至少两次，函数返回 true 。如果数组中每个元素都不
相同，则返回 false 。

示例 1:
输入: [1,2,3,1]
输出: true

示例 2:
输入: [1,2,3,4]
输出: false

示例 3:
输入: [1,1,1,3,3,4,3,2,4,2]
输出: true

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/hash-table/xhzjp6/
*/
/* 
	方法一：使用 hashMap
	思路：
		遍历数组，使用 hashMap 来记录出现过的元素，如果某个元素重复出现，
		则返回 true，否则返回 false。
	时间复杂度：O(n)
		n 是数组元素个数，我们需要遍历整个数组。
	空间复杂度：O(n)
		n 是数组元素个数，最坏情况下我们需要存储所有元素。
*/
func containsDuplicate(nums []int) bool {
	hash := make(map[int]bool)
	for _, v := range nums {
		if _, ok := hash[v]; ok {
			return true
		}
		hash[v] = true
	}
	return false
}
/* 
	方法二：排序
	思路：
		排序之后再遍历判断是否存在重复元素。
	时间复杂度：O(nlogn)
		n 是数组元素个数，我们需要先排序O(logn)后再遍历整个数组O(n)。
	空间复杂度：O(logn)
		排序时所需要的额外空间。
*/
func containsDuplicate(nums []int) bool {
	n := len(nums)
	if n < 2 {
		return false
	}
	sort.Ints(nums)
	for i := 1; i < n; i ++ {
		if nums[i] == nums[i-1] {
			return true
		}
	}
	return false
}

/* 
========================== 2、只出现一次的数字 ==========================
给定一个非空整数数组，除了某个元素只出现一次以外，其余每个元素均出现两次。找出那个只出现了一次的元素。
说明：
你的算法应该具有线性时间复杂度。 你可以不使用额外空间来实现吗？

示例 1:
输入: [2,2,1]
输出: 1

示例 2:
输入: [4,1,2,1,2]
输出: 4

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/hash-table/xhsyr2/
*/
/* 
	方法一：哈希表
	思路：
		遍历数组，使用哈希表记录元素出现的次数，最后再遍历哈希表，从中
		取出只出现一次的元素。
	时间复杂度：O(n)
		n 是数组元素个数。
	空间复杂度：O(N)
		n 是数组元素个数，我们需要记录数组元素出现的次数。
*/
func singleNumber(nums []int) int {
	hash := make(map[int]int, 0)
	for i := 0; i < len(nums); i ++ {
		hash[nums[i]] ++
	}
	for k, v := range hash {
		if v == 1 {
			return k
		}
	}
	return 0
}

/* 
	方法二：位运算
	思路：
		使用异或运算来寻找只出现过一次的数字。
		A^A = 0，0^A = A
	时间复杂度：O(n)
		n 是数组元素个数，我们需要完全遍历数组。
	时间复杂度：O(1)
		通过异或运算，我们不需要任何额外空间。
*/
func singleNumber(nums []int) int {
	resultMap := 0
	for _, v := range nums {
		resultMap ^= v
	}
	return resultMap
}

/* 
========================== 3、两个数组的交集 ==========================
给定两个数组，编写一个函数来计算它们的交集。

示例 1：
输入：nums1 = [1,2,2,1], nums2 = [2,2]
输出：[2]

示例 2：
输入：nums1 = [4,9,5], nums2 = [9,4,9,8,4]
输出：[9,4]

说明：
    输出结果中的每个元素一定是唯一的。
    我们可以不考虑输出结果的顺序。

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/hash-table/xh4sec/
*/
/* 
	方法一：哈希表法
	思路：
		先用哈希表记录长度较短的数组的元素，然后遍历另一个数组，如果
		该数组元素在哈希表中出现过，则说明它是交集元素。
	时间复杂度：O(m + n)
		m、n 分别是两个数组的元素个数，我们需要遍历两个数组进行处理。
	空间复杂度：O(min(m, n))
		m、n 分别是两个数组的元素个数，我们需要记录较短的那个数组的元素。
*/
func intersection(nums1 []int, nums2 []int) []int {
	m, n := len(nums1), len(nums2)
	hash := make(map[int]bool, 0)
	// 交集需要去重
	resultMap := make(map[int]bool, 0)
	if m < n {
		for _, v := range nums1 {
			hash[v] = true
		}
		for _, v := range nums2 {
			if _, ok := hash[v]; ok {
				resultMap[v] = true
			}
		}
	} else {
		for _, v := range nums2 {
			hash[v] = true
		}
		for _, v := range nums1 {
			if _, ok := hash[v]; ok {
				resultMap[v] = true
			}
		}
	}
	ans := make([]int, 0)
	for k, _ := range resultMap {
		ans = append(ans, k)
	}
	return ans
}

/* 
	方法二：排序+双指针
	思路：
		如果两个数组是有序的，则可以使用双指针的方法得到两个数组的交集。
		首先对两个数组进行排序，然后使用两个指针遍历两个数组。可以预见的
		是加入答案的数组的元素一定是递增的，为了保证加入元素的唯一性，我
		们需要额外记录变量 pre 表示上一次加入答案数组的元素。
		初始时，两个指针分别指向两个数组的头部。每次比较两个指针指向的两
		个数组中的数字，如果两个数字不相等，则将指向较小数字的指针右移一
		位，如果两个数字相等，且该数字不等于 pre，将该数字添加到答案并更
		新 pre 变量，同时将两个指针都右移一位。当至少有一个指针超出数组
		范围时，遍历结束。
	时间复杂度：O(mlog⁡m+nlog⁡n)
		其中 m 和 n 分别是两个数组的长度。对两个数组排序的时间复杂度分
		别是 O(mlog⁡m) 和 O(nlog⁡n)，双指针寻找交集元素的时间复杂度
		是 O(m+n)，因此总时间复杂度是 O(mlog⁡m+nlog⁡n)。
	空间复杂度：O(log⁡m+log⁡n)
		其中 m 和 n 分别是两个数组的长度。空间复杂度主要取决于排序使用的
		额外空间。
*/
func intersection(nums1 []int, nums2 []int) []int {
	sort.Ints(nums1)
	sort.Ints(nums2)
	m, n := len(nums1), len(nums2)
	i, j := 0, 0
	ans := make([]int, 0)
	for i < m && j < n {
		if nums1[i] == nums2[j] {
			// 答案中没有的数才添加进答案
			if len(ans) == 0 || nums1[i] != ans[len(ans) - 1] {
				ans = append(ans, nums1[i])
			}
			i ++
			j ++
			continue
		}
		if nums1[i] < nums2[j] {
			i ++
		} else {
			j ++
		}
	}
	return ans
}

/* 
========================== 4、快乐数 ==========================
编写一个算法来判断一个数 n 是不是快乐数。

「快乐数」定义为：
    对于一个正整数，每一次将该数替换为它每个位置上的数字的平方和。
    然后重复这个过程直到这个数变为 1，也可能是 无限循环 但始终变不到 1。
    如果 可以变为  1，那么这个数就是快乐数。

如果 n 是快乐数就返回 true ；不是，则返回 false 。

示例 1：
输入：19
输出：true
解释：
1^2 + 9^2 = 82
8^2 + 2^2 = 68
6^2 + 8^2 = 100
1^2 + 0^2 + 0^2 = 1

示例 2：
输入：n = 2
输出：false

提示：
    1 <= n <= 2^31 - 1

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/hash-table/xh1k9i/
*/
/* 
	方法一：哈希表法
	思路：
		根据快乐数的定义，当一个数的每一位上的数字的平方和存在无限循环时，
		即无法变到 1 时，它不是快乐数，我们用一个哈希表来记录每一次计算的
		平方和，当平方和重复出现时，说明该数的数位平方和是无限循环的，
		无法变到1，即不是快乐数。
	时间复杂度：O(243⋅3+logn+loglogn+logloglogn)... = O(log⁡n)
		查找给定数字的下一个值的成本为 O(log⁡n)，因为我们正在处理数字中的
		每位数字，而数字中的位数由 log⁡n 给定。
		要计算出总的时间复杂度，我们需要仔细考虑循环中有多少个数字，它们
		有多大。
		我们在上面确定，一旦一个数字低于 243，它就不可能回到 243 以上。
		因此，我们就可以用 243 以下最长循环的长度来代替 243，不过，因为
		常数无论如何都无关紧要，所以我们不会担心它。
		对于高于 243 的 n，我们需要考虑循环中每个数高于 243 的成本。通过
		数学运算，我们可以证明在最坏的情况下，这些成本将是 
			O(log⁡n)+O(log⁡log⁡n)+O(log⁡log⁡log⁡n)...
		幸运的是，O(log⁡n) 是占主导地位的部分，而其他部分相比之下都很小
		（总的来说，它们的总和小于log⁡n），所以我们可以忽略它们。
	空间复杂度：O(logn)
		与时间复杂度密切相关的是衡量我们放入哈希集合中的数字以及它们有多
		大的指标。对于足够大的 nnn，大部分空间将由 nnn 本身占用。我们可
		以很容易地优化到 O(243⋅3)=O(1)，方法是只保存集合中小于 243 的数
		字，因为对于较高的数字，无论如何都不可能返回到它们。
*/
func isHappy(n int) bool {
	hash := make(map[int]bool, 0)
	sum := n
	for sum != 1 {
		// 存在循环
		if hash[sum] {
			return false
		}
		// 标记 sum 为已处理
		hash[sum] = true
		// 重置 sum
		sum = step(sum)
	}
	return true
}
// 拆分 sum 的每一位并求平方和
func step(n int) int {
	sum := 0
	for n > 0 {
		num := (n % 10)
		n /= 10
		sum += num * num
	}
	return sum
}

/* 
	方法二：快慢指针法
	思路：
		对于不是快乐数的数，它的数位平方和总数会出现循环，而快乐数的数位
		平方和会回归到1。如此我们就可以把该题转换为类似判断链表是否有环
		的题目，有环即平方和循环，无环即可到达终点 1.
		我们定义 slow、fast 快慢指针来判断是否有环，fast 指针每次走两步，
		slow 指针每次走一步，当有环时，快指针总会追上慢指针，而无环时快指
		针会先到达终点。
	时间复杂度：O(logn)
		该分析建立在对前一种方法的分析的基础上，但是这次我们需要跟踪两个
		指针而不是一个指针来分析，以及在它们相遇前需要绕着这个循环走多少
		次。
		如果没有循环，那么快跑者将先到达 1，慢跑者将到达链表中的一半。我
		们知道最坏的情况下，成本是 O(2⋅log⁡n)=O(log⁡n)。
		一旦两个指针都在循环中，在每个循环中，快跑者将离慢跑者更近一步。
		一旦快跑者落后慢跑者一步，他们就会在下一步相遇。假设循环中有 k 个
		数字。如果他们的起点是相隔 k−1 的位置（这是他们可以开始的最远的
		距离），那么快跑者需要 k−1 步才能到达慢跑者，这对于我们的目的来
		说也是不变的。因此，主操作仍然在计算起始 n 的下一个值，即 O(log⁡n)。
	空间复杂度：O(1)
*/
func isHappy(n int) bool {
	slow, fast := n, step(n)
	for fast != 1 && fast != slow {
		fast = step(step(fast))
		slow = step(slow)
	}
	return fast == 1
}

/* 
========================== 5、两数之和 ==========================
给定一个整数数组 nums 和一个整数目标值 target，请你在该数组中找出 和为
目标值 的那 两个 整数，并返回它们的数组下标。
你可以假设每种输入只会对应一个答案。但是，数组中同一个元素在答案里不能重
复出现。
你可以按任意顺序返回答案。

示例 1：
输入：nums = [2,7,11,15], target = 9
输出：[0,1]
解释：因为 nums[0] + nums[1] == 9 ，返回 [0, 1] 。

示例 2：
输入：nums = [3,2,4], target = 6
输出：[1,2]

示例 3：
输入：nums = [3,3], target = 6
输出：[0,1]

提示：
    2 <= nums.length <= 10^3
    -109 <= nums[i] <= 10^9
    -109 <= target <= 10^9
    只会存在一个有效答案

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/hash-table/xhb0fv/
*/

/* 
	方法一：暴力解法
	思路：
		我们预先固定一个数 x ，然后再遍历数组寻找另一个数 target - x。
	时间复杂度：O(n^2)
		n 是数组元素个数。
	空间复杂度：O(1)
*/
func twoSum(nums []int, target int) []int {
	for i, x := range nums {
		for j := i + 1; j < len(nums); j ++ {
			if x + nums[j] == target && i != j {
				return []int{i, j}
			}
		}
	}
	return nil
}
/* 
	方法二：哈希表法
	思路：
		注意到方法一的时间复杂度较高的原因是寻找 target - x 的时间复杂
		度过高。因此，我们需要一种更优秀的方法，能够快速寻找数组中是否存
		在目标元素。如果存在，我们需要找出它的索引。
		使用哈希表，可以将寻找 target - x 的时间复杂度降低到从 O(N) 
		降低到 O(1)。
		这样我们创建一个哈希表，对于每一个 x，我们首先查询哈希表中是否存
		在 target - x，然后将 x 插入到哈希表中，即可保证不会让 x 和自己
		匹配。
	时间复杂度：O(n)
		n 是数组元素个数，我们需要遍历两次数组，一次是把数组元素存入哈
		希表，另一次是寻找符合要求的两个元素
	空间复杂度：O(n)
		n 是数组元素个数，我们需要把数组元素都放入哈希表中。
*/
func twoSum(nums []int, target int) []int {
	// key: nums[i], value: i
	hash := make(map[int]int, 0)
	for i, x := range nums {
		if j, ok := hash[target-x]; ok {
			// 同一个数不能使用两次
            if j != i {
				if i < j {
					return []int{i, j}
				}
				return []int{j, i}
			}
        }
        hash[x] = i
	}
	return nil
}

/* 
========================== 6、同构字符串 ==========================
给定两个字符串 s 和 t，判断它们是否是同构的。
如果 s 中的字符可以按某种映射关系替换得到 t ，那么这两个字符串是同构的。
每个出现的字符都应当映射到另一个字符，同时不改变字符的顺序。不同字符不
能映射到同一个字符上，相同字符只能映射到同一个字符上，字符可以映射到自
己本身。

示例 1:
输入：s = "egg", t = "add"
输出：true

示例 2：
输入：s = "foo", t = "bar"
输出：false

示例 3：
输入：s = "paper", t = "title"
输出：true

提示：
    可以假设 s 和 t 长度相同。

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/hash-table/xhjvbj/
*/
/* 
	方法一：哈希表法
	思路：
		此题的映射为：s 中的每一个字符在 t 中都有唯一的对应关系，t 中的
		每一个字符在 s 中也都有唯一的对应关系，即为双向映射。
		在示例2中，因为 s 中的 'o' 在 t 中对应了 'a' 和 'r'，所以不符合
		要求。
		读懂题目之后，此题就很简单了，我们可以用两个哈希表来存储 s->t 
		和 t->s 的映射关系，我们同时遍历 s 和 t，记录好 
		map[s[i]] = t[i] 和 map[t[i]] = s[i]，在遍历过程中，
		如果 map[s[i]] 或 map[t[i]]重复出现，且其映射关系与上一次不同，
		则说明 s 和 t 不是双向映射关系，不符合要求。
	时间复杂度：O(n)
		n 是字符串的长度，我们需要完整遍历 s 和 t。
	空间复杂度：O(n)
		n 是字符串的长度，我们需要记录 s->t 的映射关系。
*/
func isIsomorphic(s string, t string) bool {
	n := len(s)
	// 记录 s->t 的映射关系
	st := make(map[byte]byte, 0)
	// 记录 t->s 的映射关系
	ts := make(map[byte]byte, 0)
	for i := 0; i < n; i ++ {
		// s->t
		if v, ok := st[s[i]]; ok && v != t[i] {
			return false
		}
		// t->s
		if v, ok := ts[t[i]]; ok && v != s[i] {
			return false
		}
		st[s[i]] = t[i]
		ts[t[i]] = s[i]
	}
	return true
}

/* 
========================== 7、两个列表的最小索引总和 ==========================
假设Andy和Doris想在晚餐时选择一家餐厅，并且他们都有一个表示最喜爱餐厅的
列表，每个餐厅的名字用字符串表示。
你需要帮助他们用最少的索引和找出他们共同喜爱的餐厅。 如果答案不止一个，则
输出所有答案并且不考虑顺序。 你可以假设总是存在一个答案。

示例 1:
输入:
["Shogun", "Tapioca Express", "Burger King", "KFC"]
["Piatti", "The Grill at Torrey Pines", "Hungry Hunter Steakhouse", "Shogun"]
输出: ["Shogun"]
解释: 他们唯一共同喜爱的餐厅是“Shogun”。

示例 2:
输入:
["Shogun", "Tapioca Express", "Burger King", "KFC"]
["KFC", "Shogun", "Burger King"]
输出: ["Shogun"]
解释: 他们共同喜爱且具有最小索引和的餐厅是“Shogun”，它有最小的索引和1(0+1)。

提示:
    两个列表的长度范围都在 [1, 1000]内。
    两个列表中的字符串的长度将在[1，30]的范围内。
    下标从0开始，到列表的长度减1。
    两个列表都没有重复的元素。

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/hash-table/xhfact/
*/
/* 
	方法一：哈希表法
	思路：
		我们枚举 list1 中的每一个字符串，遍历整个 list2 一遍，对每一对字
		符串都进行比较。我们使用哈希表 map，它包含了形如 (sum:list_sum) 
		的元素。这里 sum 是匹配元素的下标和，list_sum 是下标和为 sum 的
		匹配字符串列表。
		这样，通过比较，一旦 list1 中第 i 个字符串和 list2 中第 j 个字符
		串匹配，如果 sum为 i+j 的条目在 map中还没有，我们就加一个条目。
		如果已经存在，由于我们需要保存所有下标和相同的字符串对，所以我们将
		这对字符串保存到哈希表中。
		最后我们遍历 map 的键一遍，并找到下标和最小的字符串列表。
	时间复杂度：O(l1∗l2∗x)
		list1 中的每个字符串都与 list2 中的字符串进行了比较。l1 和 l2
		是 list1 和 list2 列表的长度，x 是字符串的平均长度。
	空间复杂度：O(l1∗l2∗x)
		最坏情况下，list1 和 list2 中所有字符串都相同，那么哈希表最大会
		变成 l1∗l2∗x，其中 x 是字符串的平均长度。
*/
func findRestaurant(list1 []string, list2 []string) []string {
	m, n := len(list1), len(list2)
	hash := make(map[int][]string, 0)
	for i := 0; i < m; i ++ {
		for j := 0; j < n; j ++ {
			if list1[i] == list2[j] {
				if _, ok := hash[i + j]; !ok {
					hash[i + j] = make([]string, 0)
				}
				hash[i + j] = append(hash[i + j], list1[i])
			}
		}
	}
	minIndexSum := 1 << 31 - 1
	for key, _ := range hash {
		minIndexSum = min(minIndexSum, key)
	}
	return hash[minIndexSum]
}
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

/* 
	方法二：哈希表法（线性时间复杂度）
	思路：
		首先我们遍历 list1 一遍并为每个元素在哈希表 map 中创建一个条目，
		格式为 (list[i],i)。这里 i 是第 i 个元素的下标，list[i] 就是第 
		i 个元素本身。这样我们就创建了一个从 list1 中元素到它们下标的映
		射表。
		现在我们遍历 list2，对于每一个元素 list2[j]，我们检查在 map 中
		是否已经存在相同元素的键。如果已经存在，说明这一元素在 list1 和
		list2 中都存在。这样我们就知道了这一元素在 list1 和 list2 中的
		下标，将它们求和 sum=map.get(list[j])+j，如果这一 sum 之前记录
		的最小值要小，我们更新返回的结果列表 res，里面只保存 list2[j] 
		作为里面唯一的条目。
		如果 sum 与之前获得的最小值相等，那么我们将 list2[j] 放入结果
		列表 res。
	时间复杂度：O(l1+l2)
		list2 中的每一个字符串都会在 list1 的映射表中查找，l1​ 和 l2 分
		别是 list1 和 list2 的长度。
	空间复杂度：O(l1*x)
		hashmap 的大小为 l1*x，其中 x 是 list1 中字符串的平均长度。
*/
func findRestaurant(list1 []string, list2 []string) []string {
	hash := make(map[string]int, 0)
	for i, v := range list1 {
		hash[v] = i
	}
	res := make([]string, 0)
	min_sum := 1 << 31 - 1
	for i := 0; i < len(list2); i ++ {
		if index, ok := hash[list2[i]]; ok {
			sum := i + index
			if sum < min_sum {
				// 抛弃之前记录的值
				res = []string{list2[i]}
				min_sum = sum
			} else if sum == min_sum {
				res = append(res, list2[i])
			}
		}
	}
	return res
}

/* 
========================== 8、字符串中的第一个唯一字符 ==========================
给定一个字符串，找到它的第一个不重复的字符，并返回它的索引。如果不存在，
则返回 -1。

示例：
s = "leetcode"
返回 0

s = "loveleetcode"
返回 2

提示：你可以假定该字符串只包含小写字母。

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/hash-table/xxx94s/
*/
/* 
	方法一：使用哈希表存储频数
	思路：
		我们可以对字符串进行两次遍历。
		在第一次遍历时，我们使用哈希映射统计出字符串中每个字符出现的次数。
		在第二次遍历时，我们只要遍历到了一个只出现一次的字符，那么就返回
		它的索引，否则在遍历结束后返回 −1。
	时间复杂度：O(n)
		其中 n 是字符串 s 的长度。我们需要进行两次遍历。
	空间复杂度：O(∣Σ∣)
		其中 Σ 是字符集，在本题中 s 只包含小写字母，因此 ∣Σ∣≤26 我们需
		要 O(∣Σ∣) 的空间存储哈希映射。
*/
func firstUniqChar(s string) int {
	hash := make(map[byte]int, 0)
	for _, v := range []byte(s) {
		hash[v] ++
	}
	for i, v := range []byte(s) {
		if hash[v] == 1 {
			return i
		}
	}
	return -1
}

/* 
	方法二：使用哈希表存储索引
	思路：
		我们可以对方法一进行修改，使得第二次遍历的对象从字符串变为哈希映
		射。
		具体地，对于哈希映射中的每一个键值对，键表示一个字符，值表示它的
		首次出现的索引（如果该字符只出现一次）或者 −1（如果该字符出现多次）。当我们第一次遍历字符串时，设当前遍历到的字符为 ccc，如果 ccc 不在哈希映射中，我们就将 ccc 与它的索引作为一个键值对加入哈希映射中，否则我们将 ccc 在哈希映射中对应的值修改为 −1-1−1。
		在第一次遍历结束后，我们只需要再遍历一次哈希映射中的所有值，找出
		其中不为 −1 的最小值，即为第一个不重复字符的索引。如果哈希映射中
		的所有值均为 −1，我们就返回 −1。
	时间复杂度：O(n)
		其中 n 是字符串 s 的长度。第一次遍历字符串的时间复杂度为 O(n)，
		第二次遍历哈希映射的时间复杂度为 O(∣Σ∣)，由于 s 包含的字符种类
		数一定小于 s 的长度，因此 O(∣Σ∣) 在渐进意义下小于 O(n)，可以忽略。
	空间复杂度：O(∣Σ∣)
		其中 Σ 是字符集，在本题中 s 只包含小写字母，因此 ∣Σ∣≤26 我们需
		要 O(∣Σ∣) 的空间存储哈希映射。
*/
func firstUniqChar(s string) int {
	hash := make(map[byte]int, 0)
	for i, v := range []byte(s) {
		if _, ok := hash[v]; ok {
			hash[v] = -1
		} else {
			hash[v] = i
		}
	}
	minIndex := len(s)
	for _, v := range hash {
		if v != -1 && v < minIndex {
			minIndex = v
		}
	}
	if minIndex < len(s) {
		return minIndex
	}
	return -1
}

/* 
	方法三：使用数组
	思路：
		已知字符串的字母都是小写，那么我们就可以直接用一个长度为 26 的数组
		来记录每个字母出现的次数，最后再遍历 s，找到第一个只出现一次的那个
		字母的下标即可.
		优点：创建数组的消耗比哈希表要低很多
	时间复杂度：O(n)
		其中 n 是字符串 s 的长度。第一次遍历字符串的时间复杂度为 O(n)，
		第二次遍历字母数组的时间复杂度为 O(∣Σ∣)，由于 s 包含的字符种类
		数一定小于 s 的长度，因此 O(∣Σ∣) 在渐进意义下小于 O(n)，可以忽略。
	空间复杂度：O(∣Σ∣)
		其中 Σ 是字符集，在本题中 s 只包含小写字母，因此 ∣Σ∣≤26 我们需
		要 O(∣Σ∣) 的空间存储哈希映射。
*/
func firstUniqChar(s string) int {
	arr := make([]int, 26)
	for i := 0; i < len(s); i ++ {
		arr[s[i] - 'a'] ++
	}
	for i := 0; i < len(s); i ++  {
		if arr[s[i] - 'a'] == 1 {
			return i
		}
	}
	return -1
}

/* 
========================== 9、两个数组的交集 II ==========================
给定两个数组，编写一个函数来计算它们的交集。

示例 1：
输入：nums1 = [1,2,2,1], nums2 = [2,2]
输出：[2,2]

示例 2:
输入：nums1 = [4,9,5], nums2 = [9,4,9,8,4]
输出：[4,9]

说明：
    输出结果中每个元素出现的次数，应与元素在两个数组中出现次数的最小值一致。
    我们可以不考虑输出结果的顺序。

进阶：
    如果给定的数组已经排好序呢？你将如何优化你的算法？
    如果 nums1 的大小比 nums2 小很多，哪种方法更优？
	如果 nums2 的元素存储在磁盘上，内存是有限的，并且你不能一次加载所有
	的元素到内存中，你该怎么办？

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/hash-table/xx5hsd/
*/
/* 
	方法一：哈希表法
	思路：
		使用两个哈希表分别记录 nums1 和 nums2 中每个元素出现的次数，然后
		遍历长度较小的那个哈希表，以它的 key 为标准从两个哈希表中取出这个
		key 的出现次数，用出现次数较小且大于0的次数作为该 key 在结果集中
		出现的次数。
	时间复杂度：O(m + n)
		m、n 分别是 nums1 和 nums2 的长度，我们需要遍历两个数组进行处理。
	空间复杂度：O(m + n)
		m、n 分别是 nums1 和 nums2 的长度，我们需要记录两个数组元素出现
		的次数。
*/
func intersect(nums1 []int, nums2 []int) []int {
	h1 := make(map[int]int, 0)
	h2 := make(map[int]int, 0)
	for _, v := range nums1 {
		h1[v] ++
	}
	for _, v := range nums2 {
		h2[v] ++
	}
	if len(h1) > len(h2) {
		h1, h2 = h2, h1
	}
	res := make([]int, 0)
	for k, v := range h1 {
		cnt := h2[k]
		if cnt > v {
			cnt, v = v, cnt
		}
		if cnt > 0 {
			for i := 0; i < cnt; i ++ {
				res = append(res, k)
			}
		}
	}
	return res
}
/* 
	方法二：哈希表法（空间优化）
	思路：
		我们可以只用一个哈希表来完成。
		我们用一个哈希表记录较短数组元素出现的次数，然后遍历另一个数组，
		如果该数组元素在哈希表中出现过，则我们把它添加到结果集一次，并
		在哈希表中减少一次该元素出现的次数。
	时间复杂度：O(m + n)
	空间复杂度：O(min(m, n))
*/
func intersect(nums1 []int, nums2 []int) []int {
	hash := make(map[int]int, 0)
	if len(nums1) > len(nums2) {
		nums1, nums2 = nums2, nums1
	}
	for _, v := range nums1 {
		hash[v] ++
	}
	res := make([]int, 0)
	for _, v := range nums2 {
		if hash[v] > 0 {
			res = append(res, v)
			hash[v] --
		}
	}
	return res
}

/* 
	方法三：排序+双指针
	思路：
		排序两个数组，然后用两个指针 i, j 同时遍历两个数组，如果 
			nums1[i] < nums2[j], i ++
			nums1[i] > nums2[j], j ++
			nums1[i] == nums2[j]，i ++, j ++ 并把元素添加到结果集
	时间复杂度：O(mlogm + nlogn)
		其中 m 和 n 分别是两个数组的长度。对两个数组进行排序的时间复杂度
		是 O(mlog⁡m + nlog⁡n)，遍历两个数组的时间复杂度是 O(m + n)，
		因此总时间复杂度是 O(mlog⁡m + nlog⁡n)。
	空间复杂度：O(min⁡(m,n))
		其中 m 和 n 分别是两个数组的长度。为返回值创建一个数组 
		intersection，其长度为较短的数组的长度。
	
	结语
		如果 nums2​ 的元素存储在磁盘上，磁盘内存是有限的，并且你不能一次
		加载所有的元素到内存中。那么就无法高效地对 nums2​ 进行排序，因此
		推荐使用方法二而不是方法三。在方法二中，nums2​ 只关系到查询操作，
		因此可以每次只读取 nums2​ 中的一部分数据，并进行处理即可。
*/
func intersect(nums1 []int, nums2 []int) []int {
	sort.Ints(nums1)
	sort.Ints(nums2)
	if len(nums1) > len(nums2) {
		nums1, nums2 = nums2, nums1
	}
	res := make([]int, 0)
	i, j := 0, 0
	for i < len(nums1) && j < len(nums2) {
		if nums1[i] == nums2[j] {
			res = append(res, nums1[i])
			i ++
			j ++
			continue
		}
		if nums1[i] < nums2[j] {
			i ++
		} else {
			j ++
		}
	}
	return res
}

/* 
========================== 10、存在重复元素 II ==========================
给定一个整数数组和一个整数 k，判断数组中是否存在两个不同的索引 i 和 j，
使得 nums [i] = nums [j]，并且 i 和 j 的差的 绝对值 至多为 k。

示例 1:
输入: nums = [1,2,3,1], k = 3
输出: true

示例 2:
输入: nums = [1,0,1,1], k = 1
输出: true

示例 3:
输入: nums = [1,2,3,1,2,3], k = 2
输出: false

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/hash-table/xx5bzh/
*/
/* 
	方法一：暴力枚举【超时】
	思路：
		将每个元素与它之前的 k 个元素中比较查看它们是否相等。
	时间复杂度：O(n*min(k, n))
		每次搜索都要花费 O(min⁡(k,n)) 的时间，哪怕 k 比 n 大，
		一次搜索中也只需比较 n 次。
	空间复杂度：O(1)
*/
func containsNearbyDuplicate(nums []int, k int) bool {
	for i, v := range nums {
		for j := max(i - k, 0); j < i; j ++ {
			if v == nums[j] {
				return true
			}
		}
	}
	return false
}
func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
/* 
	方法二：哈希表法
	思路：
		维护一个长度为 k 的窗口，然后遍历数组，如果在窗口中找到当前元素，
		则返回 true，否则把当前元素添加入窗口中，如果窗口元素个数大于 k，
		则移除最新加入窗口的元素。
	时间复杂度：O(n)
		n 是数组元素个数，我们只需一次遍历数组。
	空间复杂度：O(min(n, k))
		n 是数组元素个数，开辟的额外空间取决于散列表中存储的元素的个数，
		也就是滑动窗口的大小 O(min⁡(n,k))。
*/
func containsNearbyDuplicate(nums []int, k int) bool {
	hash := make(map[int]bool)
	for i, v := range nums {
		if hash[v] {
			return true
		}
		hash[v] = true
		if len(hash) > k {
			// 移除最新加入窗口的元素
			delete(hash, nums[i - k])
		}
	}
	return false
}
