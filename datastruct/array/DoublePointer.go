package array

/* 
	双指针
*/

/* 
========================== 1、实现 strStr() ==========================
实现 strStr() 函数。
给定一个 haystack 字符串和一个 needle 字符串，在 haystack 字符串中
找出 needle 字符串出现的第一个位置 (从0开始)。如果不存在，则返回  -1。

示例 1:
输入: haystack = "hello", needle = "ll"
输出: 2

示例 2:
输入: haystack = "aaaaa", needle = "bba"
输出: -1

说明:
当 needle 是空字符串时，我们应当返回什么值呢？这是一个在面试中很好的问题。
对于本题而言，当 needle 是空字符串时我们应当返回 0 。这与C语言的 strstr() 以及 Java的 indexOf() 定义相符

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/array-and-string/cm5e2/
*/
/* 
	方法一：双指针滑动串口
	思路：
		遍历长字符串，若当前字符与短字符串的一个字符匹配，
		则从该位置开始逐一匹配长字符串与短字符串的字符，中间遇到不匹配的直接跳出。
	时间复杂度：O((m−n)n)，最优时间复杂度为 O(m)。
		其中 m 为 haystack 字符串的长度，n 为 needle 字符串的长度。
		内循环中比较字符串的复杂度为 L，总共需要比较 (m - n) 次。
	空间复杂度：O(1)。
*/
func strStr(haystack string, needle string) int {
	n, m := len(needle), len(haystack)
	if n == 0 {
		return 0
	}
	if m == 0 {
		return -1
	}
	i, j := 0, 0
	// 不需要对比完，只需要比较到 haystack 剩余长度小于等于 needle 时就行了
	for i = 0; i < m - n + 1; i ++ {
		for j = 0; j < n; j ++ {
			if haystack[i + j] != needle[j] {
				break
			}
		}
		if j == n {
			return i
		}
	}
	return -1
}

/* 
========================== 2、反转字符串 ==========================
编写一个函数，其作用是将输入的字符串反转过来。输入字符串以字符数组 char[] 的形式给出。
不要给另外的数组分配额外的空间，你必须原地修改输入数组、使用 O(1) 的额外空间解决这一问题。
你可以假设数组中的所有字符都是 ASCII 码表中的可打印字符。
 
示例 1：
输入：["h","e","l","l","o"]
输出：["o","l","l","e","h"]

示例 2：
输入：["H","a","n","n","a","h"]
输出：["h","a","n","n","a","H"]

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/array-and-string/cacxi/
*/
/* 
	方法一：双指针
	思路：
		使用收尾双指针遍历交换指向的元素。
	时间复杂度：O(n)
		其中 n 为字符数组的长度。一共执行了 n/2 次的交换。
	空间复杂度：O(1)
*/
func reverseString(s []byte)  {
	n := len(s)
	if n == 0 {
		return
	}
	L, R := 0, n - 1
	for L < R {
		s[L], s[R] = s[R], s[L]
		L ++
		R --
	}
}


/* 
========================== 3、数组拆分 I ==========================
给定长度为 2n 的整数数组 nums ，你的任务是将这些数分成 n 对, 
例如 (a1, b1), (a2, b2), ..., (an, bn) ，使得从 1 到 n 的 min(ai, bi) 总和最大。
返回该 最大总和 。

示例 1：
输入：nums = [1,4,3,2]
输出：4
解释：所有可能的分法（忽略元素顺序）为：
1. (1, 4), (2, 3) -> min(1, 4) + min(2, 3) = 1 + 2 = 3
2. (1, 3), (2, 4) -> min(1, 3) + min(2, 4) = 1 + 2 = 3
3. (1, 2), (3, 4) -> min(1, 2) + min(3, 4) = 1 + 3 = 4
所以最大总和为 4

示例 2：
输入：nums = [6,2,6,5,1,2]
输出：9
解释：最优的分法为 (2, 1), (2, 5), (6, 6). min(2, 1) + min(2, 5) + min(6, 6) = 1 + 2 + 6 = 9

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/array-and-string/c24he/
*/
/* 
	方法一：排序
	思路：
		由题意可知，要求 1 到 n 的 min(ai, bi) 的总和最大，则 min(ai, bi)
		要尽量取到最大值，也就意味着 bi - ai 的损失最小，如此，我们只需要对
		输入数据进行排序，然后按升序取 ai,bi 为一对数据，再求 min(ai,bi) 即可。
	时间复杂度：O(nlogn)
		n 是数组元素的个数，排序耗时O(logn)，计算 Sum(min(ai,bi)) 耗时 n
	空间复杂度：O(logn)
		排序排序需要使用 O(logn) 的栈空间。
*/
func arrayPairSum(nums []int) int {
	n := len(nums)
	if n == 0 {
		return 0
	}
	sort.Ints(nums)
	res := 0
	i, j := 0, 1
	for  j < n {
		res += min(nums[i], nums[j])
		i += 2
		j += 2
	}
	return res
}
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

/* 
========================== 4、两数之和 II - 输入有序数组 ==========================
给定一个已按照 升序排列  的整数数组 numbers ，请你从数组中找出两个数满足相加
之和等于目标数 target 。
函数应该以长度为 2 的整数数组的形式返回这两个数的下标值。numbers 的下标从 
1 开始计数，所以答案数组应当满足 1 <= answer[0] < answer[1] <= numbers.length 。
你可以假设每个输入只对应唯一的答案，而且你不可以重复使用相同的元素。

示例 1：
输入：numbers = [2,7,11,15], target = 9
输出：[1,2]
解释：2 与 7 之和等于目标数 9 。因此 index1 = 1, index2 = 2 。

示例 2：
输入：numbers = [2,3,4], target = 6
输出：[1,3]

示例 3：
输入：numbers = [-1,0], target = -1
输出：[1,2]

提示：
    2 <= numbers.length <= 3 * 104
    -1000 <= numbers[i] <= 1000
    numbers 按 递增顺序 排列
    -1000 <= target <= 1000
    仅存在一个有效答案

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/array-and-string/cnkjg/
*/
/* 
	方法一：二分查找
	思路：
		以 target - numbers[i] 作为 key，在 numbers[i+1 : n] 中做二分查找，
		返回 i + 1 和 二分查找有效值的下标 + 1
	时间复杂度：O(nlogn)
		以 target - numbers[i] 作为 key 有 n 个，需要遍历 n 次，而二分查找
		耗时 O(logn)，所以总的时间复杂度是 O(nlogn)
	空间复杂度：O(1)
*/
func twoSum(numbers []int, target int) []int {
	n := len(numbers)
	if n < 2 {
		return []int{-1, -1}
	}
	for i := 0; i < n; i ++ {
		key := target - numbers[i]
		L, R := i + 1, n - 1
		for L + 1 < R {
			mid := L + ((R - L ) >> 1)
			if key < numbers[mid] {
				R = mid
			} else {
				L = mid
			}
		}
		if key == numbers[L] {
			return []int{i + 1, L + 1}
		}
		if key == numbers[R] {
			return []int{i + 1, R + 1}
		}
	}
	// 找不到符合条件的数据
	return []int{-1, -1}
}

/* 
	方法二：双指针
	思路：
		初始时两个指针分别指向第一个元素位置和最后一个元素的位置。
		每次计算两个指针指向的两个元素之和，并和目标值比较。
		如果两个元素之和等于目标值，则发现了唯一解。如果两个元素之和小于目标值，
		则将左侧指针右移一位。如果两个元素之和大于目标值，则将右侧指针左移一位。
		移动指针之后，重复上述操作，直到找到答案。
	时间复杂度：O(n)
		其中 n 是数组的长度。两个指针移动的总次数最多为 n 次。
    空间复杂度：O(1)
*/
func twoSum(numbers []int, target int) []int {
	n := len(numbers)
	if n < 2 {
		return []int{-1, -1}
	}
	L, R := 0, n - 1
	for L < R {
		sum := numbers[L] + numbers[R]
		if sum == target {
			return []int{L + 1, R + 1}
		}
		if sum < target {
			L ++
		} else {
			R --
		}
	}
	return []int{-1, -1}
}

/* 
========================== 5、移除元素 ==========================
给你一个数组 nums 和一个值 val，你需要 原地 移除所有数值等于 val 的元素，
并返回移除后数组的新长度。
不要使用额外的数组空间，你必须仅使用 O(1) 额外空间并 原地 修改输入数组。
元素的顺序可以改变。你不需要考虑数组中超出新长度后面的元素。 

说明:
为什么返回数值是整数，但输出的答案是数组呢?
请注意，输入数组是以「引用」方式传递的，这意味着在函数里修改输入数组对于
调用者是可见的。
你可以想象内部操作如下:

// nums 是以“引用”方式传递的。也就是说，不对实参作任何拷贝
int len = removeElement(nums, val);
// 在函数里修改输入数组对于调用者是可见的。
// 根据你的函数返回的长度, 它会打印出数组中 该长度范围内 的所有元素。
for (int i = 0; i < len; i++) {
    print(nums[i]);
}

示例 1：
输入：nums = [3,2,2,3], val = 3
输出：2, nums = [2,2]
解释：函数应该返回新的长度 2, 并且 nums 中的前两个元素均为 2。你不需要考虑数组中超出新长度后面的元素。
例如，函数返回的新长度为 2 ，而 nums = [2,2,3,3] 或 nums = [2,2,0,0]，也会被视作正确答案。

示例 2：
输入：nums = [0,1,2,2,3,0,4,2], val = 2
输出：5, nums = [0,1,4,0,3]
解释：函数应该返回新的长度 5, 并且 nums 中的前五个元素为 0, 1, 3, 0, 4。注意这五个元素可为任意顺序。
你不需要考虑数组中超出新长度后面的元素。

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/array-and-string/cwuyj/
*/
/* 
	方法一：双指针
	思路：
		定义 slow、fast 两个指针来遍历数组，fast 指针每次走一步，而 slow 指针
		只有当 fast 指针指向的值 不等于 val 时，把 fast 指针指向的值赋给 slow，
		然后才走一步（即把有效值移动到数组的前面），最后返回 slow
	时间复杂度：O(n)
		n 是数组元素个数，两个指针最多走 n 步
	空间复杂度：O(1)
*/
func removeElement(nums []int, val int) int {
	n := len(nums)
	if n == 0 {
		return 0
	}
	slow, fast := 0, 0
	for fast < n {
		if nums[fast] != val {
			nums[slow] = nums[fast]
			slow ++
		}
		fast ++
	}
	return slow
}

/* 
	方法二：双指针 —— 当要删除的元素很少时
	思路：
		现在考虑数组包含很少的要删除的元素的情况。例如，num=[1，2，3，5，4]，Val=4。
		之前的算法会对前四个元素做不必要的复制操作。另一个例子是 num=[4，1，2，3，5]，Val=4。
		似乎没有必要将 [1，2，3，5] 这几个元素左移一步，因为问题描述中提到元素的顺序可以更改。
	算法：
		当我们遇到 nums[i]=val 时，我们可以将当前元素与最后一个元素进行交换，
		并释放最后一个元素。这实际上使数组的大小减少了 1。
		请注意，被交换的最后一个元素可能是您想要移除的值。
		但是不要担心，在下一次迭代中，我们仍然会检查这个元素。
	时间复杂度：O(n)
		i 和 n 最多遍历 n 步。在这个方法中，赋值操作的次数等于要删除的元素的数量。
		因此，如果要移除的元素很少，效率会更高。
	空间复杂度：O(1)
*/
func removeElement(nums []int, val int) int {
	i, n := 0, len(nums)
	if n == 0 {
		return 0
	}
	for i < n {
		if nums[i] == val {
			// 把最后一个元素放到当前位置
			nums[i] = nums[n - 1]
			// 释放最后一个元素
			n --
		} else {
			// 因为最后一个元素也有可能时要被删除的，所以不能跳过，
			// 必须用 else 继续对比 i 位置
			i ++
		}
	}
	return i
}

/* 
========================== 6、最大连续1的个数 ==========================
给定一个二进制数组， 计算其中最大连续 1 的个数。

示例：
输入：[1,1,0,1,1,1]
输出：3
解释：开头的两位和最后的三位都是连续 1 ，所以最大连续 1 的个数是 3.

提示：
    输入的数组只包含 0 和 1 。
    输入数组的长度是正整数，且不超过 10,000。

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/array-and-string/cd71t/
*/
/* 
	方法一：一次遍历
	算法：
		用一个计数器 count 记录 1 的数量，另一个计数器 maxCount 记录当前最大的 1 的数量。
		当我们遇到 1 时，count 加一。
		当我们遇到 0 时：
			将 count 与 maxCount 比较，maxCoiunt 记录较大值。
			将 count 设为 0。
		返回 maxCount。
	时间复杂度：O(N)
		N 值得是数组的长度。
	空间复杂度：O(1)
		仅仅使用了 count 和 maxCount。
*/
func findMaxConsecutiveOnes(nums []int) int {
	ln := len(nums)
	if ln == 0 {
		return 0
	}
	cnt, maxCount := 0, 0
	for i := 0; i < ln; i ++ {
		if nums[i] == 1 {
			cnt ++
		} else {
			maxCount = max(maxCount, cnt)
			cnt = 0
		}
	}
	return max(maxCount, cnt)
}
/* 
	方法二：双指针
	思路：
		让 left、right指针同时指向第一个1，之后遇到 1 时 right ++，
		遇到 0 时计算 right - left，即为当前连续1的个数，再与最大值比较和更新最大值。
		然后从 right 之后重置 left、right指针到新的 1，重复上述过程找出最大的连续1。
	时间复杂度：O(N)
		N 值得是数组的长度。
	空间复杂度：O(1)
*/
func findMaxConsecutiveOnes(nums []int) int {
	n := len(nums)
	if n == 0 {
		return 0
	}
	left := findLeft(nums, 0)
	right := left
	ans := 0
	for right < n {
		if nums[right] != 1 {
			ans = max(ans, right - left)
			// 从 right 的下一个开始找
			left = findLeft(nums, right + 1)
			right = left
		} else {
			right ++
		}
	}
	// 当最后面是连续的 1 时需要计算
	return max(ans, right - left)
}
func findLeft(nums []int, left int) int {
	for left < len(nums) {
		if nums[left] == 1 {
			break
		}
		left ++
	}
	return left
}
func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

/* 
========================== 7、长度最小的子数组 ==========================
给定一个含有 n 个正整数的数组和一个正整数 target 。
找出该数组中满足其和 ≥ target 的长度最小的 连续子数
组 [numsl, numsl+1, ..., numsr-1, numsr] ，并返回其长度。
如果不存在符合条件的子数组，返回 0 。

示例 1：
输入：target = 7, nums = [2,3,1,2,4,3]
输出：2
解释：子数组 [4,3] 是该条件下的长度最小的子数组。

示例 2：
输入：target = 4, nums = [1,4,4]
输出：1

示例 3：
输入：target = 11, nums = [1,1,1,1,1,1,1,1]
输出：0

提示：
    1 <= target <= 109
    1 <= nums.length <= 105
    1 <= nums[i] <= 105

进阶：
	如果你已经实现 O(n) 时间复杂度的解法, 请尝试设计一个 O(n log(n))
	 时间复杂度的解法。

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/array-and-string/c0w4r/
*/
/* 
	方法一：滑动窗口算法
	思路：
		用左右指针来标记窗口的起始位置和结束位置，维护 winSum 表示窗口内元素的
		和，然后窗口向右扩展尝试达到目标条件，即窗口内元素的和大于等于 target，
		当窗口内元素符合条件时，在符合条件下尝试收缩窗口，并更新最小窗口大小，
		最后返回最小窗口的大小。
	时间复杂度：O(n)
		n 表示数组元素的个数，最坏情况下，左右指针需要各遍历一次数组，即当
		数组的最后一个元素满足最小连续子数组和条件时。
	空间复杂度：O(1)
*/
func minSubArrayLen(target int, nums []int) int {
	n := len(nums)
	if n == 0 {
		return 0
	}
	// 窗口左右边界
	L, R := 0, 0
	// 窗口内元素的和
	winSum := 0
	// 最小窗口长度
	min := (1 << 63) - 1
	for R < n {
		// 窗口扩展
		winSum += nums[R]
		for winSum >= target {
			// 记录最小窗口长度
			if R - L + 1 < min {
				min = R - L + 1
			}
			// 窗口收缩
			winSum -= nums[L]
			L ++
		}
		R ++
	}
	if min == (1 << 63) - 1 {
		return 0
	}
	// 返回最小窗口长度
	return min
}

/* 
========================== 8、删除排序数组中的重复项 ==========================
给定一个排序数组，你需要在 原地 删除重复出现的元素，使得每个元素只出现一次，
返回移除后数组的新长度。
不要使用额外的数组空间，你必须在 原地 修改输入数组 并在使用 O(1) 额外空间
的条件下完成。

示例 1:
给定数组 nums = [1,1,2], 
函数应该返回新的长度 2, 并且原数组 nums 的前两个元素被修改为 1, 2。 
你不需要考虑数组中超出新长度后面的元素。

示例 2:
给定 nums = [0,0,1,1,1,2,2,3,3,4],
函数应该返回新的长度 5, 并且原数组 nums 的前五个元素被修改为 0, 1, 2, 3, 4。
你不需要考虑数组中超出新长度后面的元素。

说明:
为什么返回数值是整数，但输出的答案是数组呢?
请注意，输入数组是以「引用」方式传递的，这意味着在函数里修改输入数组对于调用者是可见的。
你可以想象内部操作如下:
// nums 是以“引用”方式传递的。也就是说，不对实参做任何拷贝
int len = removeDuplicates(nums);
// 在函数里修改输入数组对于调用者是可见的。
// 根据你的函数返回的长度, 它会打印出数组中该长度范围内的所有元素。
for (int i = 0; i < len; i++) {
    print(nums[i]);
}

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/array-and-string/cq376/
*/
/* 
	方法一：双指针
	思路：
		因为数组是排序的，所有重复元素都是相邻着。
		使用 slow、fast 两个指针遍历数组，fast 指针每次走一步，slow 指针
		只有当 fast 指向的值不等于 slow 指针指向的值的时候，slow 指针才走
		一步，并交换两个指针指向的值。
	时间复杂度：O(n)
		n 是数组的元素个数，两个指针最多走 n 步
	空间复杂度：O(1)
*/
func removeDuplicates(nums []int) int {
	n := len(nums)
	slow, fast := 0, 0
	for fast < n {
		if nums[fast] != nums[slow] {
			slow ++
			nums[slow], nums[fast] = nums[fast], nums[slow]
		}
		fast ++
	}
	// 因为 slow 是从 0 开始的，作为长度返回时需要 + 1
	return slow + 1
}

/* 
========================== 9、移动零 ==========================
给定一个数组 nums，编写一个函数将所有 0 移动到数组的末尾，同时保持非零元素的相对顺序。
示例:
输入: [0,1,0,3,12]
输出: [1,3,12,0,0]

说明:
    必须在原数组上操作，不能拷贝额外的数组。
    尽量减少操作次数

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/array-and-string/c6u02/
*/
/* 
/* 
	方法一：双指针
	思路：
		左右指针初始化在数组头部，右指针循环右移，每当右指针指向非零时，
		交换左右指针指向的元素，同时左指针右移。
	注意：
		此方法中同时指向非零时，它们的位置是一致的，即指向相同元素，
		只会和自己交换而不存在非零元素被移到 0 后面的情况。
		因此每次交换，都是将左指针的零与右指针的非零数交换，且非零数的相对顺序并未改变。
	缺点：
		假如数组中全是 0，那么我们也需要交换 n 次，造成了不必要的交换性能损失。
	时间复杂度：O(n)
		其中 n 为序列长度。每个位置至多被遍历两次。
	空间复杂度：O(1)
		只需要常数的空间存放若干变量。
*/
func moveZeroes(nums []int)  {
	n := len(nums)
	if n == 0 {
		return
	}
	L, R := 0, 0
	for R < n {
		if nums[R] != 0 {
			nums[L], nums[R] = nums[R], nums[L]
			L ++
		}
		R ++
	}
}

/* 
	方法二：双指针【优化】
	思路：
		左右指针初始化在数组头部，当左指针指向非零时，左右指针同时右移。
		当左指针指向零且右指针指向非零时，交换左右指针指向的元素，同时左指针右移。
		否则只移动右指针
	优化：
		此方法只有当左指针指向0，右指针指向非0时才进行交换，相对于方法一减少了
		对于 0 的交换次数。
	时间复杂度：O(n)
		其中 n 为序列长度。每个位置至多被遍历两次。
	空间复杂度：O(1)
		只需要常数的空间存放若干变量。
*/
func moveZeroes(nums []int)  {
	n := len(nums)
	if n == 0 {
		return
	}
	L, R := 0, 0
	for R < n {
        // 左指针指向非0，左右指针同时右移
        if nums[L] != 0 {
            L ++
            R ++
            continue
        }
        // 左指针指向0时，如果右指针指向非0，则交换且移动左指针，
		// 否则只移动右指针
        if nums[R] != 0 {
			nums[L], nums[R] = nums[R], nums[L]
			L ++
		}
		R ++
	}
}