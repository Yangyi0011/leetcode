package binarysearch

/* 
	二分查找中的常见问题
*/
/* 
===================== 1、Pow(x, n) =====================
实现 pow(x, n) ，即计算 x 的 n 次幂函数（即，xn）。

示例 1：
输入：x = 2.00000, n = 10
输出：1024.00000

示例 2：
输入：x = 2.10000, n = 3
输出：9.26100

示例 3：
输入：x = 2.00000, n = -2
输出：0.25000
解释：2-2 = 1/22 = 1/4 = 0.25

提示：
    -100.0 < x < 100.0
    -2^31 <= n <= 2^31-1
    -10^4 <= xn <= 10^4

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/binary-search/xe7k32/
*/
/* 
	方法一：快速幂 + 递归
	思路：
		「快速幂算法」的本质是分治算法。举个例子，如果我们要计算 x^64，我们可以按
		照：
			x → x^2 → x^4 → x^8 → x^16 → x^32 → x^64

		的顺序，从 x 开始，每次直接把上一次的结果进行平方，计算 6 次就可以得到 
		x^64 的值，而不需要对 x 乘 63 次 x。

		再举一个例子，如果我们要计算 x^77，我们可以按照：
			x → x^2 → x^4 → x^9 → x^19 → x^38 → x^77
		的顺序，在 x → x^2，x^2 → x^4，x^19 → x^38 这些步骤中，我们直接把上一次
		的结果进行平方，而在 x^4 → x^9，x^9 → x^19，x^38 → x^77 这些步骤中，
		我们把上一次的结果进行平方后，还要额外乘一个 x。

		直接从左到右进行推导看上去很困难，因为在每一步中，我们不知道在将上一次的
		结果平方之后，还需不需要额外乘 x。但如果我们从右往左看，分治的思想就十分
		明显了：
			当我们要计算 x^n 时，我们可以先递归地计算出 y = x^(⌊n/2⌋)，其中 ⌊a⌋ 表示
			对 a 进行下取整；
			根据递归计算的结果，如果 n 为偶数，那么 x^n = y^2；如果 n 为奇数，
			那么 x^n = y^2*x；
			递归的边界为 n = 0，任意数的 0 次方均为 1。

		由于每次递归都会使得指数减少一半，因此递归的层数为 O(log⁡n)，算法可以在很
		快的时间内得到结果。
	时间复杂度：O(logn)
		即为递归的层数。
	空间复杂度：O(logn)
		即为递归的层数，是递归所需的栈空间。
*/
func myPow(x float64, n int) float64 {
	var quickMul func(x float64, n int) float64
	quickMul = func(x float64, n int) float64 {
		if n == 0 {
			return 1
		}
		y := quickMul(x, n/2)
		if n % 2 == 1 {
			return y*y*x
		}
		return y*y
	}
	if n >= 0 {
		return quickMul(x, n)
	}
	// 处理负次幂，即对其正数幂求倒数
	return 1.0 / quickMul(x, -n)
}

/* 
	方法二：快速幂 + 迭代
	思路：
		由于递归需要使用额外的栈空间，我们试着将递归转写为迭代。在方法一中，
		我们也提到过，从左到右进行推导是不容易的，因为我们不知道是否需要额外
		乘 x。但我们不妨找一找规律，看看哪些地方额外乘了 x，并且它们对答案产
		生了什么影响。
		我们还是以 x^77 作为例子：
			x → x^2 → x^4→ +x^9 → +x^19 → x^38 → +x^77

		并且把需要额外乘 x 的步骤打上了 + 标记。可以发现：
			x^38 → +x^77 中额外乘的 x 在 x^77 中贡献了 x；
			x^9 → +x^19 中额外乘的 x 在之后被平方了 2 次，因此在 x^77 中
			贡献了 (x^2)^2 = x^4；
			x^4 → +x^9 中额外乘的 x 在之后被平方了 3 次，因此在 x^77 中
			贡献了 (x^2)^3 = x^8；
			最初的 x 在之后被平方了 6 次，因此在 x^77 中贡献了 
			(x^2)^6 = x^64;

		我们把这些贡献相乘，x*x^4*x^8*x^64 恰好等于 x^77。而这些贡献的指数
		部分又是什么呢？它们都是 2 的幂次，这是因为每个额外乘的 x 在之后都
		会被平方若干次。而这些指数 1，4，8 和 64，恰好就对应了 77 的二进制
		表示 (10011010)2​ 中的每个 1！

		因此我们借助整数的二进制拆分，就可以得到迭代计算的方法，一般地，如果
		整数 n 的二进制拆分为
			n = 2^(i0) + 2^(i1) + ⋯ + 2^(ik)
		那么
			x^n = (x^2)^(i0) * (x^2)^(i1) * ⋯ * (x^2)^(ik)

		这样以来，我们从 x 开始不断地进行平方，得到 x2,x4,x8,x16,⋯，如果 
		n 的第 k 个（从右往左，从 0 开始计数）二进制位为 1，那么我们就将对
		应的贡献 (x^2)^k 计入答案。
*/
func myPow(x float64, n int) float64 {
	var quickMul func(x float64, n int) float64
	quickMul = func(x float64, n int) float64 {
		ans := 1.0
		// 贡献的初始值为 x
		x_contribute := x
		// 在对 n 进行二进制拆分的同时计算答案
		for n > 0 {
			if n % 2 == 1 {
				// 如果 n 二进制表示的最低位为 1，那么需要计入贡献
				ans *= x_contribute
			}
			// 将贡献不断地平方
			x_contribute *= x_contribute
			// 舍弃 n 二进制表示的最低位，这样我们每次只要判断最低位即可
			n >>= 1
		}
		return ans
	}
	if n >= 0 {
		return quickMul(x, n)
	}
	// 处理负次幂，即对其正数幂求倒数
	return 1.0 / quickMul(x, -n)
}

/* 
	方法三：快速幂【迭代优化】
	思路：
		快速幂算法：
			核心思想是每一步都把指数减半，而相应的底数做平方运算。
		我们以 3^10 作为例子：
			3^10 = 3*3*3*3*3*3*3*3*3*3
			3^10 = (3*3)^5
			3^10 = 9^5						// 指数减半，而底数平方
			3^10 = 9^4 * 9^1				// 把奇数指数-1变成成偶数，分离出其底数
			3^10 = 81^2 * 9^1				// 继续偶数指数减半，偶数底数平方，已经分离出的底数不动
			3^10 = 6561^1 * 9^1				// 重复偶数指数减半，偶数底数平方，奇数指数-1并分离出其底数
			3^10 = 6561^0 * 6561^1 * 9^1	// 当指数为 0 时结束
		由此可见：
			最后求出的幂结果实际上就是在“指数减半，底数平方”的变化过程中，
			所有的当指数为奇数时所分离出的底数的乘积。
	时间复杂度：O(logn)
	空间复杂度：O(1)
*/
func myPow(x float64, n int) float64 {
	var quickMul func(x float64, n int) float64
	quickMul = func(x float64, n int) float64 {
		// 结果
		ans := 1.0
		for n > 0 {
			// 指数为奇数时，需要把分离出来的底数乘入结果
			// 同 n % 2 == 1
			if n & 1 == 1 {
				ans *= x
			}
			// 指数减半，同 n = n / 2
			n >>= 1
			// 底数平方
			x *= x
		}
		return ans
	}
	if n >= 0 {
		return quickMul(x, n)
	}
	// 处理负数幂，即对其正数幂求倒数
	return 1.0 / quickMul(x, -n)
}

/* 
===================== 2、有效的完全平方数 =====================
给定一个 正整数 num ，编写一个函数，如果 num 是一个完全平方数，则返回 true ，否则
返回 false 。
进阶：不要 使用任何内置的库函数，如  sqrt 。

示例 1：
输入：num = 16
输出：true

示例 2：
输入：num = 14
输出：false

提示：
    1 <= num <= 2^31 - 1

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/binary-search/xel3tc/
*/
/* 
	方法一：二分查找【模板三】
	思路：
		在每一次二分过程中对比 mid*mid 和 num 的大小，如果：
			mid*mid > num，则往左找
			否则往右找
	时间复杂度：O(lognum)
	空间复杂度：O(1)
*/
func isPerfectSquare(num int) bool {
	if num < 2 {
		return true
	}
	L, R := 1, num
	for L + 1 < R {
		mid := L + ((R - L) >> 1)
		if mid * mid > num {
			R = mid
		} else {
			L = mid
		}
	}
	if L * L == num {
		return true
	}
	if R * R == num {
		return true
	}
	return false
}

/* 
===================== 3、寻找比目标字母大的最小字母 =====================
给你一个排序后的字符列表 letters ，列表中只包含小写英文字母。另给出一个目标字母 
target，请你寻找在这一有序列表里比目标字母大的最小字母。
在比较时，字母是依序循环出现的。举个例子：
	如果目标字母 target = 'z' 并且字符列表为 letters = ['a', 'b']，则答案返
	回 'a'

示例：
输入:
letters = ["c", "f", "j"]
target = "a"
输出: "c"

输入:
letters = ["c", "f", "j"]
target = "c"
输出: "f"

输入:
letters = ["c", "f", "j"]
target = "d"
输出: "f"

输入:
letters = ["c", "f", "j"]
target = "g"
输出: "j"

输入:
letters = ["c", "f", "j"]
target = "j"
输出: "c"

输入:
letters = ["c", "f", "j"]
target = "k"
输出: "c"

提示：
    letters长度范围在[2, 10000]区间内。
    letters 仅由小写字母组成，最少包含两个不同的字母。
    目标字母target 是一个小写字母。

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/binary-search/xeiuui/
*/
/* 
	方法一：二分查找【模板二】
	思路：
		使用二分查找去查找 target 的下一个字母，如果不存在则返回第一个
	时间复杂度：O(logn)
		n 为有序字符列表的长度。
	空间复杂度：O(1)
*/
func nextGreatestLetter(letters []byte, target byte) byte {
	n := len(letters)
	L, R := 0, n
	target += 1
	for L < R {
		mid := L + ((R - L ) >> 1)
		if target > letters[mid] {
			L = mid + 1
		} else {
			R = mid
		}
	}
	// 找不到返回第一个字母
	if L == n {
		return letters[0]
	}
	return letters[L]
}

/* 
===================== 4、寻找重复数 =====================
给定一个包含 n + 1 个整数的数组 nums ，其数字都在 1 到 n 之间（包括 1 和 n），
可知至少存在一个重复的整数。
假设 nums 只有 一个重复的整数 ，找出 这个重复的数 。

示例 1：
输入：nums = [1,3,4,2,2]
输出：2

示例 2：
输入：nums = [3,1,3,4,2]
输出：3

示例 3：
输入：nums = [1,1]
输出：1

示例 4：
输入：nums = [1,1,2]
输出：1

提示：
    2 <= n <= 3 * 10^4
    nums.length == n + 1
    1 <= nums[i] <= n
	nums 中 只有一个整数 出现 两次或多次 ，其余整数均只出现 一次
进阶：

    如何证明 nums 中至少存在一个重复的数字?
    你可以在不修改数组 nums 的情况下解决这个问题吗？
    你可以只用常量级 O(1) 的额外空间解决这个问题吗？
    你可以设计一个时间复杂度小于 O(n^2) 的解决方案吗？

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/binary-search/xe6xnr/
*/
/* 
	方法一：哈希表法
	思路：
		遍历数组，使用哈希表记录元素是否出现过，出现过则说明该元素重复。
	时间复杂度：O(n)
	空间复杂度：O(n)
*/
func findDuplicate(nums []int) int {
	hash := make(map[int]bool, 0)
	for _, v := range nums {
		if hash[v] {
			return v
		}
		hash[v] = true
	}
	return 0
}
/* 
	方法二：排序 + 二分查找【模板二】
	思路：
		先对数组进行排序，再遍历排序后的数组，以 nums[i] 为 key，对剩余数组 
		nums[i+1, n) 进行二分查找，找到则说明该元素是重复元素。
	时间复杂度：O(nlogn)
		n 是数组长度，排序的复杂度是 O(nlogn)，排序之后进行二分查找的复杂度是
		O(nlogn)，所以总的时间复杂度是 O(nlogn)。
	空间复杂的：O(logn)
		快速排序需要 O(logn) 的额外空间。
*/
func findDuplicate(nums []int) int {
	sort.Ints(nums)
	for i, v := range nums {
		if binarySearch(nums, v, i + 1, len(nums)) != -1 {
			return v
		}
	}
	return 0
}
func binarySearch(nums []int, target, L, R int) int {
	for L < R {
		mid := L + ((R - L) >> 1)
		if target > nums[mid] {
			L = mid + 1
		} else {
			R = mid
		}
	}
	if L < len(nums) && nums[L] == target {
		return L
	}
	return -1
}

/* 
	方法三：快慢指针
	思路：
		我们对 nums 数组建图，每个位置 i 连一条 i→nums[i] 的边。由于存在的重复的
		数字 target，因此 target 这个位置一定有起码两条指向它的边，因此整张图一定
		存在环，且我们要找到的 target 就是这个环的入口，那么整个问题就等价于 
			142. 环形链表 II。
			https://leetcode-cn.com/problems/linked-list-cycle-ii/solution/
		
		step1: 快慢指针都在起点
			  f|s
			   ↓
		value：1 4 6 6 6 2 3
		index：0 1 2 3 4 5 6

		step2:	s 走一步指向了第 nums[0] 个节点，即 index=1，f 走两步指向了第
				nums[nums[0]] 个节点，即 index = 4
			     s     f
			     ↓     ↓
		value：1 4 6 6 6 2 3
		index：0 1 2 3 4 5 6

		step3:	s 走一步指向了 index = 4，f 走两步指向了 index = 3，此时 
				value[s] == value[f]，即 f 与 s 相遇
			         f s
			         ↓ ↓
		value：1 4 6 6 6 2 3
		index：0 1 2 3 4 5 6

		step4:	f 回到起点 index = 0，s 不动 index = 4
			   f       s
			   ↓       ↓
		value：1 4 6 6 6 2 3
		index：0 1 2 3 4 5 6

		step5:	s 走一步指向 index = 6，f 走一步指向 index = 1
			     f         s
			     ↓         ↓
		value：1 4 6 6 6 2 3
		index：0 1 2 3 4 5 6

		step6:	s 走一步指向 index = 3，f 走一步指向 index = 4，此时
				value[s] == value[f]，即 f 与 s 再次相遇，返回 6
			         s f
			         ↓ ↓
		value：1 4 6 6 6 2 3
		index：0 1 2 3 4 5 6

		我们先设置慢指针 slow 和快指针 fast ，慢指针每次走一步，快指针每次走两步，
		根据「Floyd 判圈算法」两个指针在有环的情况下一定会相遇，此时我们再将 slow 
		放置起点 0，两个指针每次同时移动一步，相遇的点就是答案。
	时间复杂度：O(n)
		「Floyd 判圈算法」时间复杂度为线性的时间复杂度。
	空间复杂度：O(1)
		我们只需要常数空间存放若干变量。
*/
func findDuplicate(nums []int) int {
    slow, fast := 0, 0
    for slow, fast = nums[slow], nums[nums[fast]]; slow != fast; slow, fast = nums[slow], nums[nums[fast]] { }
    slow = 0
    for slow != fast {
        slow = nums[slow]
        fast = nums[fast]
    }
    return slow
}

/* 
===================== 5、寻找两个正序数组的中位数 =====================
给定两个大小分别为 m 和 n 的正序（从小到大）数组 nums1 和 nums2。请你找出并返回
这两个正序数组的 中位数 。

示例 1：
输入：nums1 = [1,3], nums2 = [2]
输出：2.00000
解释：合并数组 = [1,2,3] ，中位数 2

示例 2：
输入：nums1 = [1,2], nums2 = [3,4]
输出：2.50000
解释：合并数组 = [1,2,3,4] ，中位数 (2 + 3) / 2 = 2.5

示例 3：
输入：nums1 = [0,0], nums2 = [0,0]
输出：0.00000

示例 4：
输入：nums1 = [], nums2 = [1]
输出：1.00000

示例 5：
输入：nums1 = [2], nums2 = []
输出：2.00000

提示：
    nums1.length == m
    nums2.length == n
    0 <= m <= 1000
    0 <= n <= 1000
    1 <= m + n <= 2000
    -10^6 <= nums1[i], nums2[i] <= 10^6

进阶：你能设计一个时间复杂度为 O(log (m+n)) 的算法解决此问题吗？

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/binary-search/xe6jas/
*/
/* 
	方法一：双指针
	思路：
		预先获得两有序数组合并后的长度 n，以及合并数组的中位数可能存在的下标：
			如果 n 是奇数，则中位数下标为 n/2，下标从 0 开始，
				中位数即为合并数组 nums[n/2]
			如果 n 是偶数，则中位数下标 n/2-1 和 n/2，下标从 0 开始，
				中位数即为合并数组 (nums[n/2-1] + nums[(n/2])/2.0
		我们使用 i、j 双指针同时遍历两个数组，遍历过程中谁小就先合并谁，但我们并
		不需要记录合并数组的每一个元素，我们只需要记录符合中位数下标的元素即可，也
		就是说，不是中位数下标的元素我们会在合并过程中直接丢弃，当所需的中位数个数
		记录完毕后直接返回。
	时间复杂度：O(m+n)
		m、n 分别是 nums1、nums2 的长度，我们需要在遍历过程中进行合并数组操作，
		合并到中位数所处的位置。
	空间复杂度：O(1)
*/
func findMedianSortedArrays(nums1 []int, nums2 []int) float64 {
	n1, n2 := len(nums1), len(nums2)
	n := n1 + n2
	// 如果 n 是偶数，则中位数下标有两个，一个是 n/2-1，另一个是 n/2
	mid := make([]int, 0)
	i, j := 0, 0
	for i < n1 && j < n2 {
		if nums1[i] < nums2[j] {
			if n & 1 == 0 && i + j == (n >> 1) - 1 {
				mid = append(mid, nums1[i])
			}
			if i + j == n >> 1 {
				mid = append(mid, nums1[i])
                i ++
				break
			}
			i ++
		} else {
			if n & 1 == 0 && i + j == (n >> 1) - 1 {
				mid = append(mid, nums2[j])
			}
			if i + j == n >> 1 {
				mid = append(mid, nums2[j])
                j ++
				break
			}
			j ++
		}
	}
	// 处理较长数组的剩余元素
	for i < n1 {
		if n & 1 == 0 && i + j == (n >> 1) - 1 {
			mid = append(mid, nums1[i])
		}
		if i + j == n >> 1 {
			mid = append(mid, nums1[i])
			break
		}
		i ++
	}
	for j < n2 {
		if n & 1 == 0 && i + j == (n >> 1) - 1 {
			mid = append(mid, nums2[j])
		}
		if i + j == n >> 1 {
			mid = append(mid, nums2[j])
			break
		}
		j ++
	}
	sum := 0
	for _, v := range mid {
		sum += v
	}
	// n 为偶数，中位数为两数之和
	if n & 1 == 0 {
		return float64(sum) / 2.0
	}
	return float64(sum)
}

/* 
	方法二：二分查找
	思路：
		根据中位数的定义，当 m+n 是奇数时，中位数是两个有序数组中的第 (m+n)/2 个元
		素，当 m+n是偶数时，中位数是两个有序数组中的第 (m+n)/2 个元素和第 (m+n)/2+1
		个元素的平均值。因此，这道题可以转化成寻找两个有序数组中的第 k 小的数，其中
		k 为 (m+n)/2 或 (m+n)/2+1。

		假设两个有序数组分别是 A。要找到第 k 个元素，我们可以比较 A[k/2−1] 和 
		B[k/2−1]，其中 / 表示整数除法。由于 A[k/2−1] 和 B[k/2−1] 的前面分别有 
		A[0 .. k/2−2] 和 B[0 .. k/2−2]，即 k/2−1 个元素，对于 A[k/2−1] 和 
		B[k/2−1] 中的较小值，最多只会有 (k/2−1)+(k/2−1)≤k−2 个元素比它小，那么它
		就不能是第 k 小的数了。

		因此我们可以归纳出三种情况：
			如果 A[k/2−1] < B[k/2−1]，则比 A[k/2−1] 小的数最多只有 A 的前 
				k/2−1 个数和 B 的前 k/2−1 个数，即比 A[k/2−1] 小的数最多只
				有 k−2 个，因此 A[k/2−1] 不可能是第 k 个数，A[0] 到 A[k/2−1]
				也都不可能是第 k 个数，可以全部排除。
			如果 A[k/2−1] > B[k/2−1]，则可以排除 B[0] 到 B[k/2−1]。
			如果 A[k/2−1] = B[k/2−1]，则可以归入第一种情况处理。

		可以看到，比较 A[k/2−1] 和 B[k/2−1] 之后，可以排除 k/2 个不可能是第 k 小
		的数，查找范围缩小了一半。同时，我们将在排除后的新数组上继续进行二分查找，
		并且根据我们排除数的个数，减少 k 的值，这是因为我们排除的数都不大于第 k 小
		的数。

		有以下三种情况需要特殊处理：
			如果 A[k/2−1] 或者 B[k/2−1] 越界，那么我们可以选取对应数组中的最后
				一个元素。在这种情况下，我们必须根据排除数的个数减少 k 的值，而
				不能直接将 k 减去 k/2。
			如果一个数组为空，说明该数组中的所有元素都被排除，我们可以直接返回另
			一个数组中第 k 小的元素。
			如果 k=1，我们只要返回两个数组首元素的最小值即可。

		用一个例子说明上述算法。假设两个有序数组如下：
			A: 1 3 4 9
			B: 1 2 3 4 5 6 7 8 9
		两个有序数组的长度分别是 4 和 9，长度之和是 13，中位数是两个有序数组中的
		第 7 个元素，因此需要找到第 k=7 个元素。

		比较两个有序数组中下标为 k/2−1=2 的数，即 A[2] 和 B[2]，如下面所示：
			A: 1 3 4 9
				   ↑
			B: 1 2 3 4 5 6 7 8 9
				   ↑
		由于 A[2] > B[2]，因此排除 B[0] 到 B[2]，即数组 B 的下标偏移（offset）
		变为 3，同时更新 k 的值：k = k−k/2 = 4

		下一步寻找，比较两个有序数组中下标为 k/2−1=1 的数，即 A[1] 和 B[4]，如下
		面所示，其中方括号部分表示已经被排除的数。
			A: 1 3 4 9
				 ↑
			B: [1 2 3] 4 5 6 7 8 9
						 ↑
		由于 A[1] < B[4]，因此排除 A[0] 到 A[1]，即数组 A 的下标偏移变为 2，同时
		更新 k 的值：k = k−k/2 = 2

		下一步寻找，比较两个有序数组中下标为 k/2−1=0 的数，即比较 A[2] 和 B[3]，
		如下面所示，其中方括号部分表示已经被排除的数。
			A: [1 3] 4 9
					 ↑
			B: [1 2 3] 4 5 6 7 8 9
					   ↑
		由于 A[2]=B[3]，根据之前的规则，排除 A 中的元素，因此排除 A[2]，即数组 A 
		的下标偏移变为 3，同时更新 k 的值： k = k−k/2 = 1

		由于 k 的值变成 1，因此比较两个有序数组中的未排除下标范围内的第一个数，其
		中较小的数即为第 k 个数，由于 A[3] > B[3]，因此第 k 个数是 B[3]=4。
			A: [1 3 4] 9
					   ↑
			B: [1 2 3] 4 5 6 7 8 9

	时间复杂度：O(log(m+n))
		其中 m 和 n 分别是数组 nums1​ 和 nums2​ 的长度。初始时有 k=(m+n)/2 或 
		k=(m+n)/2+1，每一轮循环可以将查找范围减少一半，因此时间复杂度是 O(log⁡(m+n))
	空间复杂度：O(1)
*/
func findMedianSortedArrays(nums1 []int, nums2 []int) float64 {
	// 获取总长度
	totalLength := len(nums1) + len(nums2)
	// 如果长度为奇数，说明只有一个中位数，否则有两个
    if totalLength & 1 == 1 {
        midIndex := totalLength >> 1
        return float64(getKthElement(nums1, nums2, midIndex + 1))
    } else {
        midIndex1, midIndex2 := (totalLength >> 1) - 1, totalLength >> 1
        return float64(getKthElement(nums1, nums2, midIndex1 + 1) + getKthElement(nums1, nums2, midIndex2 + 1)) / 2.0
    }
    return 0
}
// 二分查找两个有序数组中的 第 k 小的数
func getKthElement(nums1, nums2 []int, k int) int {
    index1, index2 := 0, 0
    for {
		// 如果其中一个数组为空，则返回另一个数组的第 k 位
        if index1 == len(nums1) {
            return nums2[index2 + k - 1]
        }
        if index2 == len(nums2) {
            return nums1[index1 + k - 1]
		}
		// 如果 k == 1，返回两个数组的第一位数中较小的那个
        if k == 1 {
            return min(nums1[index1], nums2[index2])
		}
		// 二分，计算新的下标
        half := k/2
        newIndex1 := min(index1 + half, len(nums1)) - 1
		newIndex2 := min(index2 + half, len(nums2)) - 1
		// 获取新下标对应的值
		pivot1, pivot2 := nums1[newIndex1], nums2[newIndex2]
		// 如果 A[k/2−1] < B[k/2−1]，说明 A[0 : k/2] 不可能是第 k 小的数，舍去
		// 并更新 k 的值
        if pivot1 <= pivot2 {
            k -= (newIndex1 - index1 + 1)
            index1 = newIndex1 + 1
        } else {
            k -= (newIndex2 - index2 + 1)
            index2 = newIndex2 + 1
        }
    }
    return 0
}
func min(x, y int) int {
    if x < y {
        return x
    }
    return y
}