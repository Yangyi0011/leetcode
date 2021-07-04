package hash

/* 
========================== 1、设计哈希集合 ==========================
不使用任何内建的哈希表库设计一个哈希集合（HashSet）。
实现 MyHashSet 类：
    void add(key) 向哈希集合中插入值 key 。
    bool contains(key) 返回哈希集合中是否存在这个值 key 。
	void remove(key) 将给定值 key 从哈希集合中删除。如果哈希集合中没有
	这个值，什么也不做。

示例：
输入：
["MyHashSet", "add", "add", "contains", "contains", "add", 
"contains", "remove", "contains"]
[[], [1], [2], [1], [3], [2], [2], [2], [2]]
输出：
[null, null, null, true, false, null, true, null, false]

解释：
MyHashSet myHashSet = new MyHashSet();
myHashSet.add(1);      // set = [1]
myHashSet.add(2);      // set = [1, 2]
myHashSet.contains(1); // 返回 True
myHashSet.contains(3); // 返回 False ，（未找到）
myHashSet.add(2);      // set = [1, 2]
myHashSet.contains(2); // 返回 True
myHashSet.remove(2);   // set = [1]
myHashSet.contains(2); // 返回 False ，（已移除）

提示：
    0 <= key <= 10^6
    最多调用 10^4 次 add、remove 和 contains 。

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/hash-table/xh377h/
*/
/* 
	方法一：采用拉链法来处理冲突
	思路：
		定义哈希函数、桶和链表，采用哈希函数来确定 key 会被存
		放到哪一个桶中，如果桶中元素冲突，则用拉链法新增元素添加到桶中
		链表的尾部。
	时间复杂度：O(n/b)
		其中 n 为哈希表中的元素数量，b 为链表的数量。假设哈希值是均匀分
		布的，则每个链表大概长度为 n/b​。
		单次操作的时间复杂度是 O(1)，但是极端情况下所有元素都被分配到
		同一个桶中时，哈希表中的所有元素将形成一条链，此时的操作时间复杂度
		将上升到 O(n)。
	空间复杂度：O(n+b)
*/
type Node struct {
	Val int
	Next *Node
}
type MyHashSet struct {
	Capacity int
	Buckets []*Node
}

/** Initialize your data structure here. */
func Constructor() MyHashSet {
	prime := prevPrime(10000)
	return MyHashSet{
		// 采用素数来做容量能有效提高内存利用率和避免冲突
		Capacity: prime,
		Buckets : make([]*Node, prime),
	}
}
// 判断一个数是不是素数
func isPrime(n int) bool {
	if n < 3 {
		return n > 1
	}
	sqrt := int(math.Floor(math.Sqrt(float64(n))))
	for i := 2; i <= sqrt; i ++ {
		if n % i == 0 {
			return false
		}
	}
	return true
}

// 获取指定数字的上一个素数
func prevPrime(n int) int {
	for i := n - 1; i >= 2; i -- {
		if isPrime(i) {
			return i
		}
	}
	return 2
}

// 哈希函数
func (this *MyHashSet) HashFunction(key int) (index int){
	return key % this.Capacity
}

// 添加元素
func (this *MyHashSet) Add(key int)  {
	// 元素已存在则不作处理
	if this.Contains(key) {
		return
	}
	node := &Node{
		Val: key,
	}
	index := this.HashFunction(key)
	head := this.Buckets[index]
	if head == nil {
		this.Buckets[index] = node
		return
	}
	for head.Next != nil {
		head = head.Next
	}
	// 桶中没有 key 这个元素，需要添加
	head.Next = node
}

func (this *MyHashSet) Remove(key int)  {
	// 元素不存在则不作处理
	if !this.Contains(key) {
		return
	}
	index := this.HashFunction(key)
	head := this.Buckets[index]
	if head.Val == key {
		this.Buckets[index] = head.Next
		head.Next = nil
		return
	}
	// 找到目标元素
	for head.Next.Val != key {
		head = head.Next
	}
	// 删除目标元素
	p := head.Next
	head.Next = head.Next.Next
	p.Next = nil
}

/** Returns true if this set contains the specified element */
func (this *MyHashSet) Contains(key int) bool {
	// 分配桶下标
	index := this.HashFunction(key)
	// 获取桶内链表的表头
	head := this.Buckets[index]
	if head == nil {
		return false
	}
	for head != nil {
		if head.Val == key {
			return true
		}
		head = head.Next
	}
	return false
}

/**
 * Your MyHashSet object will be instantiated and called as such:
 * obj := Constructor();
 * obj.Add(key);
 * obj.Remove(key);
 * param_3 := obj.Contains(key);
 */