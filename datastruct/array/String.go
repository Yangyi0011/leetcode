package array

/*
	字符串
*/

/*
========================== 1、最长公共前缀 ==========================
编写一个函数来查找字符串数组中的最长公共前缀。
如果不存在公共前缀，返回空字符串 ""。

示例 1：
输入：strs = ["flower","flow","flight"]
输出："fl"

示例 2：
输入：strs = ["dog","racecar","car"]
输出：""
解释：输入不存在公共前缀。

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/array-and-string/ceda1/
*/
/*
	方法一：纵向对比
	思路：
		以第一个元素为基准，对比每一个字符串的字符，发现不同直接返回。
	时间复杂度：O(kn)
		k 表示数组中最短字符串的长度，n 表示数组字符串的个数
	空间复杂度：O(k)
		k 表示数组中最短字符串的长度
*/
func longestCommonPrefix(strs []string) string {
	n := len(strs)
	if n == 0 {
		return ""
	}
	for i := 0; i < len(strs[0]); i++ {
		for j := 1; j < len(strs); j++ {
			// 注意下标不要越界
			if i == len(strs[j]) || strs[0][i] != strs[j][i] {
				return strs[0][:i]
			}
		}
	}
	return strs[0]
}

/*
	方法二：分治法
	思路：
		最长公共前缀满足结合律：LCP(S1​…Sn​)=LCP(LCP(S1​…Sk​),LCP(Sk+1​…Sn​))
		其中 LCP(S1…Sn) 是字符串 S1…Sn​ 的最长公共前缀，1<k<n。
		基于上述结论，可以使用分治法得到字符串数组中的最长公共前缀。
		对于问题 LCP(Si⋯Sj)，可以分解成两个子问题 LCP(Si…Smid) 与 LCP(Smid+1…Sj)，
		其中 mid=(i+j)/2​。对两个子问题分别求解，然后对两个子问题的解计算最长公共前缀，即为原问题的解。
	时间复杂度：O(mn)
		其中 m 是字符串数组中的字符串的平均长度，n 是字符串的数量。
		时间复杂度的递推式是 T(n)=2⋅T(n/2)+O(m)，通过计算可得 T(n)=O(mn)。
	空间复杂度：O(mlog⁡n)
		其中 m 是字符串数组中的字符串的平均长度，n 是字符串的数量。
		空间复杂度主要取决于递归调用的层数，层数最大为 log⁡n，每层需要 m 的空间存储返回结果。
*/
func longestCommonPrefix(strs []string) string {
	if len(strs) == 0 {
		return ""
	}
	var lcp func(int, int) string
	lcp = func(start, end int) string {
		// 开始==结束说明只有一个元素
		if start == end {
			return strs[start]
		}
		// divide
		mid := (start + end) >> 1
		lcpLeft := lcp(start, mid)
		lcpRight := lcp(mid+1, end)

		// conquer
		minLen := min(len(lcpLeft), len(lcpRight))
		for i := 0; i < minLen; i++ {
			if lcpLeft[i] != lcpRight[i] {
				return lcpLeft[:i]
			}
		}
		return lcpLeft[:minLen]
	}
	return lcp(0, len(strs)-1)
}
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

/*
========================== 2、最长回文子串 ==========================
给你一个字符串 s，找到 s 中最长的回文子串。

示例 1：
输入：s = "babad"
输出："bab"
解释："aba" 同样是符合题意的答案。

示例 2：
输入：s = "cbbd"
输出："bb"

示例 3：
输入：s = "a"
输出："a"

示例 4：
输入：s = "ac"
输出："a"

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/array-and-string/conm7/
*/

/*
方法一：动态规划
	思路与算法:
		对于一个子串而言，如果它是回文串，并且长度大于 2，那么将它首尾的两个字母去除之后，它仍然是个回文串。
		例如对于字符串 “ababa”，如果我们已经知道 “bab”是回文串，那么 “ababa” 一定是回文串，
		这是因为它的首尾两个字母都是 “a”。
		根据这样的思路，我们就可以用动态规划的方法解决本题。我们用 P(i,j) 表示字符串 s 的第 i 到 j
		个字母组成的串（下文表示成 s[i:j]）是否为回文串：
			P(i,j)= true，子串 Si…Sj 是回文串
					false， 其它情况​
		这里的「其它情况」包含两种可能性：
			（1）s[i,j] 本身不是一个回文串；
			（2）i>j，此时 s[i,j] 本身不合法。
		那么我们就可以写出动态规划的状态转移方程：
			P(i,j)=P(i+1,j−1)∧(Si==Sj)，^表示并
		也就是说，只有 s[i+1:j−1] 是回文串，并且 s 的第 i 和 j 个字母相同时，s[i:j] 才会是回文串。

		上文的所有讨论是建立在子串长度大于 2 的前提之上的，我们还需要考虑动态规划中的边界条件，
		即子串的长度为 1 或 2。对于长度为 1 的子串，它显然是个回文串；对于长度为 2 的子串，
		只要它的两个字母相同，它就是一个回文串。
		因此我们就可以写出动态规划的边界条件：
			P(i,i)=true
			P(i,i+1)=(Si​==Si+1​)​
		根据这个思路，我们就可以完成动态规划了，最终的答案即为所有
			P(i,j)=true 中 j−i+1（即子串长度）的最大值。
		注意：在状态转移方程中，我们是从长度较短的字符串向长度较长的字符串进行转移的，
			因此一定要注意动态规划的循环顺序。
	时间复杂度：O(n^2)
		其中 n 是字符串的长度。动态规划的状态总数为 O(n^2)，对于每个状态，我们需要转移的时间为 O(1)。
	空间复杂度：O(n^2)
		即存储动态规划状态需要的空间。
*/
func longestPalindrome(s string) string {
	n := len(s)
	if n == 0 || n == 1 {
		return s
	}
	// dp[i][j] 记录 s[i:j] 是不是回文的状态
	dp := make([][]bool, n)
	for i := 0; i < n; i++ {
		dp[i] = make([]bool, n)
	}
	// 记录最终结果在原字符串中的起始位置和终止位置（包含）
	// 只记录位置而不是直接截取字符串可以减少性能消耗
	start, end := 0, 0
	// sl 表示当前子串的长度，从 0 开始计数方便 i、j 下标的变化
	for sl := 0; sl < n; sl++ {
		// i 表示子串的起始下标，j 表示子串的终止下标
		for i := 0; i+sl < n; i++ {
			j := i + sl
			if sl == 0 {
				// 只有一个字符
				dp[i][j] = true
			} else if sl == 1 {
				// 有两个字符
				if s[i] == s[j] {
					dp[i][j] = true
				}
			} else {
				// 三个及以上的字符
				if s[i] == s[j] && dp[i+1][j-1] {
					dp[i][j] = true
				}
			}
			// 如果当前子串是回文子串，且长度大于已记录的最长回文子串，则更新
			// 结果的起始位置和终止位置，此处 end - start 不加1是因为 sl 也是从 [0,n) 的
			if dp[i][j] && sl > (end-start) {
				start, end = i, j
			}
		}
	}
	return s[start : end+1]
}

/*
	方法二：中心扩展算法
	思路与算法：
		我们仔细观察一下方法一中的状态转移方程：
			一个元素：P(i,i)=true
			两个元素：P(i,i+1)=(Si==Si+1)
			三个以上：P(i,j)=P(i+1,j−1)∧(Si==Sj)
		找出其中的状态转移链：
			P(i,j)←P(i+1,j−1)←P(i+2,j−2)←⋯←某一边界情况
		可以发现，所有的状态在转移的时候的可能性都是唯一的。也就是说，
		我们可以从每一种边界情况开始「扩展」，也可以得出所有的状态对应的答案。
		边界情况即为子串长度为 1 或 2 的情况。我们枚举每一种边界情况，并从对应的子串开始不断地向两边扩展。
			如果两边的字母相同，我们就可以继续扩展，例如从 P(i+1,j−1) 扩展到 P(i,j)；
			如果两边的字母不同，我们就可以停止扩展，因为在这之后的子串都不可能是回文串了。
		「边界情况」对应的子串实际上就是我们「扩展」出的回文串的「回文中心」。

		方法二的本质即为：我们枚举所有的「回文中心」并尝试「扩展」，直到无法扩展为止，
		此时的回文串长度即为此「回文中心」下的最长回文串长度。
		我们对所有的长度求出最大值，即可得到最终的答案。
	时间复杂度：O(n^2)
		其中 n 是字符串的长度。长度为 1 和 2 的回文中心分别有 n 和 n−1 个，每个回文中心最多会向外扩展 O(n) 次。
	空间复杂度：O(1)
*/
func longestPalindrome(s string) string {
	n := len(s)
	if n == 0 {
		return s
	}
	start, end := 0, 0
	for i := 0; i < n; i++ {
		// 一个字符
		left1, right1 := expandAroundCenter(s, i, i)
		// 两个字符
		left2, right2 := expandAroundCenter(s, i, i+1)

		if right1-left1 > end-start {
			start, end = left1, right1
		}
		if right2-left2 > end-start {
			start, end = left2, right2
		}
	}
	return s[start : end+1]
}

// 中心扩散
func expandAroundCenter(s string, left, right int) (int, int) {
	for left >= 0 && right < len(s) && s[left] == s[right] {
		left--
		right++
	}
	// 退出循环的时 left、right 会有一个超出限制条件，
	// 所以实际值应当按变化条件回退一步
	return left + 1, right - 1
}

/*
========================== 4、反转字符串中的单词 III ==========================
给定一个字符串，你需要反转字符串中每个单词的字符顺序，同时仍保留空格和单词
的初始顺序。

示例：
输入："Let's take LeetCode contest"
输出："s'teL ekat edoCteeL tsetnoc"

提示：
    在字符串中，每个单词由单个空格分隔，并且字符串中不会有任何额外的空格。

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/array-and-string/c8su7/
*/
/*
	方法一：双指针
	思路：
		使用 L、R 指针标记每一个单词的起始位置和终止位置，如此得出一个单词
		s[L:R]，然后对单词 s[L:R] 进行逐一字符反转，反转后 L = R + 1（指向
		下一个单词的第一个字符，因为一个单词只由单个空格分隔），如此一直处理
		直到所有单词都被反转。
	时间复杂度：O(nk)
		n 是字符串长度，k 是字符串中所有单词的平均长度
	空间复杂度：O(n)
*/
func reverseWords(s string) string {
	n := len(s)
	if n == 0 {
		return s
	}
	// 把 s 转为 byte 以方便反转操作
	bytes := []byte(s)
	L, R := 0, 0
	for R < n {
		if bytes[R] == ' ' {
			// 直接对 bytes 进行原地反转
			reverseBytes(bytes[L:R])
			// 越过单词间的空格指向下一个单词的第一个字符
			L = R + 1
		}
		R++
	}
	// 处理最后一个单词
	reverseBytes(bytes[L:R])
	return string(bytes)
}

// 原地反转 byte 数组
func reverseBytes(bytes []byte) {
	n := len(bytes)
	if n == 0 {
		return
	}
	for i := 0; i < (n >> 1); i++ {
		bytes[i], bytes[n-1-i] = bytes[n-1-i], bytes[i]
	}
}