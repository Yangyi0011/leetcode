package hash

/* 
========================== 1、设计哈希映射 ==========================
不使用任何内建的哈希表库设计一个哈希映射（HashMap）。
实现 MyHashMap 类：
    MyHashMap() 用空映射初始化对象
	void put(int key, int value) 向 HashMap 插入一个键值对 
	(key, value) 。如果 key 已经存在于映射中，则更新其对应的值 value 。
	int get(int key) 返回特定的 key 所映射的 value ；如果映射中不包含 
	key 的映射，返回 -1 。
	void remove(key) 如果映射中存在 key 的映射，则移除 key 和它所对应
	的 value 。

示例：
输入：
["MyHashMap", "put", "put", "get", "get", "put", "get", "remove", "get"]
[[], [1, 1], [2, 2], [1], [3], [2, 1], [2], [2], [2]]
输出：
[null, null, null, 1, -1, null, 1, null, -1]

解释：
MyHashMap myHashMap = new MyHashMap();
myHashMap.put(1, 1); // myHashMap 现在为 [[1,1]]
myHashMap.put(2, 2); // myHashMap 现在为 [[1,1], [2,2]]
myHashMap.get(1);    // 返回 1 ，myHashMap 现在为 [[1,1], [2,2]]
myHashMap.get(3);    // 返回 -1（未找到），myHashMap 现在为 [[1,1], [2,2]]
myHashMap.put(2, 1); // myHashMap 现在为 [[1,1], [2,1]]（更新已有的值）
myHashMap.get(2);    // 返回 1 ，myHashMap 现在为 [[1,1], [2,1]]
myHashMap.remove(2); // 删除键为 2 的数据，myHashMap 现在为 [[1,1]]
myHashMap.get(2);    // 返回 -1（未找到），myHashMap 现在为 [[1,1]]

提示：
    0 <= key, value <= 10^6
    最多调用 104 次 put、get 和 remove 方法

作者：力扣 (LeetCode)
链接：https://leetcode-cn.com/leetbook/read/hash-table/xhqwd3/
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
	Key int
	Val int
	Next *Node
}

type MyHashMap struct {
	Capacity int
	Buckets []*Node
}

/** Initialize your data structure here. */
func Constructor() MyHashMap {
	prime := 9973
	return MyHashMap{
		Capacity: prime,
		Buckets: make([]*Node, prime),
	}
}

/** value will always be non-negative. */
func (this *MyHashMap) Put(key int, value int)  {
	index := this.HashFunction(key)
	head := this.Buckets[index]
	// 存在则更新
	if this.HasKey(key) {
		for head.Key != key {
			head = head.Next
		}
		head.Val = value
		return
	}
	// 不存在则添加
	node := &Node{
		Key: key,
		Val: value,
	}
	if head == nil {
		this.Buckets[index] = node
		return
	}
	for head.Next != nil {
		head = head.Next
	}
	head.Next = node
}

/** Returns the value to which the specified key is mapped, or -1 if this map contains no mapping for the key */
func (this *MyHashMap) Get(key int) int {
	if !this.HasKey(key) {
		return -1
	}
	index := this.HashFunction(key)
	head := this.Buckets[index]
	for head.Key != key {
		head = head.Next
	}
	return head.Val
}

/** Removes the mapping of the specified value key if this map contains a mapping for the key */
func (this *MyHashMap) Remove(key int)  {
	if !this.HasKey(key) {
		return
	}
	index := this.HashFunction(key)
	head := this.Buckets[index]
	if head.Key == key {
		this.Buckets[index] = head.Next
		head.Next = nil
		return
	}
	for head.Next.Key != key {
		head = head.Next
	}
	p := head.Next
	head.Next = head.Next.Next
	p.Next = nil
}
// 哈希函数
func (this *MyHashMap) HashFunction(key int) (index int){
	return key % this.Capacity
}

// 查看是否包含某个 key
func (this *MyHashMap) HasKey(key int) bool {
	// 分配桶下标
	index := this.HashFunction(key)
	// 获取桶内链表的表头
	head := this.Buckets[index]
	if head == nil {
		return false
	}
	for head != nil {
		if head.Key == key {
			return true
		}
		head = head.Next
	}
	return false
}

/**
 * Your MyHashMap object will be instantiated and called as such:
 * obj := Constructor();
 * obj.Put(key,value);
 * param_2 := obj.Get(key);
 * obj.Remove(key);
 */