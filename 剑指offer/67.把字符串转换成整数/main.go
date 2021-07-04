package main

import "fmt"

/*
============== 剑指 Offer 67. 把字符串转换成整数 ==============

写一个函数 StrToInt，实现把字符串转换成整数这个功能。不能使用 atoi 或者其他类似
的库函数。

首先，该函数会根据需要丢弃无用的开头空格字符，直到寻找到第一个非空格的字符为止。

当我们寻找到的第一个非空字符为正或者负号时，则将该符号与之后面尽可能多的连续数字组
合起来，作为该整数的正负号；假如第一个非空字符是数字，则直接将其与之后连续的数字字
符组合起来，形成整数。

该字符串除了有效的整数部分之后也可能会存在多余的字符，这些字符可以被忽略，它们对于函
数不应该造成影响。

注意：假如该字符串中的第一个非空格字符不是一个有效整数字符、字符串为空或字符串仅包含
空白字符时，则你的函数不需要进行转换。

在任何情况下，若函数不能进行有效的转换时，请返回 0。

说明：
假设我们的环境只能存储 32 位大小的有符号整数，那么其数值范围为 [−2^31,  2^31 − 1]。
如果数值超过这个范围，请返回  INT_MAX (231 − 1) 或 INT_MIN (−231) 。

示例 1:
输入: "42"
输出: 42

示例 2:
输入: "   -42"
输出: -42
解释: 第一个非空白字符为 '-', 它是一个负号。
     我们尽可能将负号与后面所有连续出现的数字组合起来，最后得到 -42 。

示例 3:
输入: "4193 with words"
输出: 4193
解释: 转换截止于数字 '3' ，因为它的下一个字符不为数字。

示例 4:
输入: "words and 987"
输出: 0
解释: 第一个非空字符是 'w', 但它不是数字或正、负号。
     因此无法执行有效的转换。

示例 5:
输入: "-91283472332"
输出: -2147483648
解释: 数字 "-91283472332" 超过 32 位有符号整数范围。
     因此返回 INT_MIN (−2^31) 。

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/ba-zi-fu-chuan-zhuan-huan-cheng-zheng-shu-lcof
*/
/*
	方法一：遍历
	思路：
		遍历字符串，对遇到的每一个字符逐一处理，在此我们需要对一些特殊字符做处理：
			1、对于空格，过滤掉数字之前的空格，把数字之后的空格都当做非法字符。
			2、对于正负号，只记录数字之前的一个正负号，把多个正负号和数字之后的正负号
				都当做非法字符。
			3、对于'0' ~ '9' 以外的字符，都当做非法字符。
		遇到合法字符则把它转为数字并加到结果中，遇到非法字符，直接跳出循环。
		越界处理：
			1、需要在处理过程中去判断当前结果是否越界，而不是在最后去处理，因为正数
				越界后会得到一个负数的结果，在最后去处理越界问题只会得到一个负数。
			2、处理越界时需要带入正负号，因为我们在越界之前所计算的数字是正数，只会
				产生正数越界，如果字符串是'-'开头，那么越界时不带入正负号，只在最后
				返回之前处理正负号的话，将会导致我们的计算结果是 -2147483647，
				而不是正确的 -2147483648
	时间复杂度：O(n)
		n 是字符串长度，最坏情况下所有字符都是合法字符，此时我们需要处理每一个字符。
	空间复杂度：O(1)
*/
func strToInt(str string) int {
	n := len(str)
	if n == 0 {
		return 0
	}
	min := -1 << 31
	max := (1 << 31) - 1
	// 记录正负号
	var sign byte = '0'
	// 记录结果
	ans := 0
	// 标记是否已经处理过数字或正负号，正负号与数字之间的特殊符号视为非法
	flag := false
	for i := 0; i < n; i++ {
		// 处理数字之前的空格
		if str[i] == ' ' {
			// 跳过数字之前的空格
			if !flag {
				continue
			}
			// 数字之后的空格视为非法
			break
		}
		// 处理数字之前的正负号
		if str[i] == '+' || str[i] == '-' {
			// 只处理数字开头的第一个正负号，后面出现的视为非法
			if !flag && sign == '0' {
				sign = str[i]
				flag = true
				continue
			}
			break
		}
		// 处理非法字符
		if str[i] < '0' || str[i] > '9' {
			break
		}
		// 把合法字符转为数字
		ans = ans*10 + int(str[i]-'0')
		flag = true
		// 处理越界
		if ans > max {
			if sign == '-' {
				return min
			}
			return max
		}
	}
	// 处理正负号
	if sign == '-' {
		ans = -ans
	}
	return ans
}

func main() {
	str := "  -0012a42"
	res := strToInt(str)
	fmt.Println(res)
}
