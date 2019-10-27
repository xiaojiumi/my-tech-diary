# 树

## 递归

一棵树要么是空树，要么有两个指针，每个指针指向一棵树。树是一种递归结构，很多树的问题可以使用递归来处理。

### 14. 找出二叉树中第二小的节点

[671. Second Minimum Node In a Binary Tree (Easy)](https://leetcode.com/problems/second-minimum-node-in-a-binary-tree/description/)

```
Input:
   2
  / \
 2   5
    / \
    5  7

Output: 5
```

一个节点要么具有 0 个或 2 个子节点，如果有子节点，那么根节点是最小的节点。

```java
//根据题意，这题可以转换成求，左右子节点中的最小值．

class Solution {
    public int findSecondMinimumValue(TreeNode root) {
     return traversal(root,root.val);
    }
    
    private int traversal(TreeNode root,int value){
        if(root == null){
            return -1;
        }
        if(root.val > value){
            return root.val;
        }
        // 寻找左右子节点中，第一个大于自己的节点
        int l = traversal(root.left,value);
        int r = traversal(root.right,value);

        // 存在两个子节点
        if(l>=0 && r>=0){
            return Math.min(l,r);
        }
        //　存在0个子节点
        return Math.max(l,r);
    }
  
}
```

# 层次遍历

使用 BFS 进行层次遍历。不需要使用两个队列来分别存储当前层的节点和下一层的节点，因为在开始遍历一层的节点时，当前队列中的节点数就是当前层的节点数，只要控制遍历这么多节点数，就能保证这次遍历的都是当前层的节点。

## 1. 一棵树每层节点的平均数

[637. Average of Levels in Binary Tree (Easy)](https://leetcode.com/problems/average-of-levels-in-binary-tree/description/)

```
public List<Double> averageOfLevels(TreeNode root) {
    List<Double> ret = new ArrayList<>();
    if (root == null) return ret;
    Queue<TreeNode> queue = new LinkedList<>();
    queue.add(root);
    while (!queue.isEmpty()) {
        int cnt = queue.size();
        double sum = 0;
        for (int i = 0; i < cnt; i++) {
            TreeNode node = queue.poll();
            sum += node.val;
            if (node.left != null) queue.add(node.left);
            if (node.right != null) queue.add(node.right);
        }
        ret.add(sum / cnt);
    }
    return ret;
}
```

# 前中后序遍历

## 1. 非递归实现二叉树的前序遍历

[144. Binary Tree Preorder Traversal (Medium)](https://leetcode.com/problems/binary-tree-preorder-traversal/description/)

```
public List<Integer> preorderTraversal(TreeNode root) {
    List<Integer> ret = new ArrayList<>();
    Stack<TreeNode> stack = new Stack<>();
    stack.push(root);
    while (!stack.isEmpty()) {
        TreeNode node = stack.pop();
        if (node == null) continue;
        ret.add(node.val);
        stack.push(node.right);  // 先右后左，保证左子树先遍历
        stack.push(node.left);
    }
    return ret;
}
```

## 2. 非递归实现二叉树的后序遍历

[145. Binary Tree Postorder Traversal (Medium)](https://leetcode.com/problems/binary-tree-postorder-traversal/description/)

前序遍历为 root -> left -> right，后序遍历为 left -> right -> root。可以修改前序遍历成为 root -> right -> left，那么这个顺序就和后序遍历正好相反。

```
public List<Integer> postorderTraversal(TreeNode root) {
    List<Integer> ret = new ArrayList<>();
    Stack<TreeNode> stack = new Stack<>();
    stack.push(root);
    while (!stack.isEmpty()) {
        TreeNode node = stack.pop();
        if (node == null) continue;
        ret.add(node.val);
        stack.push(node.left);
        stack.push(node.right);
    }
    Collections.reverse(ret);
    return ret;
}
```

## 3. 非递归实现二叉树的中序遍历

[94. Binary Tree Inorder Traversal (Medium)](https://leetcode.com/problems/binary-tree-inorder-traversal/description/)

![img](https://pic.leetcode-cn.com/e2b24c9eb433f8ba14c44891e11371dcb9bd8ee9f89d958802cfb502da7a94c2-image.png)

```
public List<Integer> inorderTraversal(TreeNode root) {
    List<Integer> ret = new ArrayList<>();
    if (root == null) return ret;
    Stack<TreeNode> stack = new Stack<>();
    TreeNode cur = root;
    while (cur != null || !stack.isEmpty()) {
        while (cur != null) {
            stack.push(cur);
            cur = cur.left;
        }
        TreeNode node = stack.pop();
        ret.add(node.val);
        cur = node.right;
    }
    return ret;
}
```

# BST

二叉查找树（BST）：根节点大于等于左子树所有节点，小于等于右子树所有节点。

二叉查找树中序遍历有序。

## 2. 寻找二叉查找树的第 k 个元素

[230. Kth Smallest Element in a BST (Medium)](https://leetcode.com/problems/kth-smallest-element-in-a-bst/description/)

中序遍历解法：

```
private int cnt = 0;
private int val;

public int kthSmallest(TreeNode root, int k) {
    inOrder(root, k);
    return val;
}

private void inOrder(TreeNode node, int k) {
    if (node == null) return;
    inOrder(node.left, k);
    cnt++;
    if (cnt == k) {
        val = node.val;
        return;
    }
    inOrder(node.right, k);
}
```

递归解法：

```
public int kthSmallest(TreeNode root, int k) {
    int leftCnt = count(root.left);
    if (leftCnt == k - 1) return root.val;
    if (leftCnt > k - 1) return kthSmallest(root.left, k);
    return kthSmallest(root.right, k - leftCnt - 1);
}

private int count(TreeNode node) {
    if (node == null) return 0;
    return 1 + count(node.left) + count(node.right);
}
```

## 3. 把二叉查找树每个节点的值都加上比它大的节点的值

[Convert BST to Greater Tree (Easy)](https://leetcode.com/problems/convert-bst-to-greater-tree/description/)

```
Input: The root of a Binary Search Tree like this:

              5
            /   \
           2     13

Output: The root of a Greater Tree like this:

             18
            /   \
          20     13
```

先遍历右子树。

```
private int sum = 0;

public TreeNode convertBST(TreeNode root) {
    traver(root);
    return root;
}

private void traver(TreeNode node) {
    if (node == null) return;
    traver(node.right);
    sum += node.val;
    node.val = sum;
    traver(node.left);
}
```

## 4. 二叉查找树的最近公共祖先

[235. Lowest Common Ancestor of a Binary Search Tree (Easy)](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree/description/)

方法一 （递归）
思路

节点 pp，qq 的最近公共祖先（LCA）是距离这两个节点最近的公共祖先节点。在这里 最近 考虑的是节点的深度。下面这张图能帮助你更好的理解 最近 这个词的含义。

![image.png](https://pic.leetcode-cn.com/d4ae198f46c063a84c91dc0488917f7d501d2ee352a398aec9d7e3f7ecd97fc2-image.png)

笔记：pp 和 qq 其中的一个在 LCA 节点的左子树上，另一个在 LCA 节点的右子树上。

也有可能是下面这种情况：

![image.png](https://pic.leetcode-cn.com/57065f75de837616260b54bac172ef9b3c1afab04468ef2ea3394a6708949fa7-image.png)

算法

从根节点开始遍历树
如果节点 pp 和节点 qq 都在右子树上，那么以右孩子为根节点继续 1 的操作
如果节点 pp 和节点 qq 都在左子树上，那么以左孩子为根节点继续 1 的操作
如果条件 2 和条件 3 都不成立，这就意味着我们已经找到节 pp 和节点 qq 的 LCA 了



```
        _______6______
      /                \
  ___2__             ___8__
 /      \           /      \
0        4         7        9
        /  \
       3   5

For example, the lowest common ancestor (LCA) of nodes 2 and 8 is 6. Another example is LCA of nodes 2 and 4 is 2, since a node can be a descendant of itself according to the LCA definition.
```

```java
public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
    if (root.val > p.val && root.val > q.val) return lowestCommonAncestor(root.left, p, q);
    if (root.val < p.val && root.val < q.val) return lowestCommonAncestor(root.right, p, q);
    return root;
}
```

## 5. 二叉树的最近公共祖先

[236. Lowest Common Ancestor of a Binary Tree (Medium)](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/description/)

```
       _______3______
      /              \
  ___5__           ___1__
 /      \         /      \
6        2       0        8
        /  \
       7    4

For example, the lowest common ancestor (LCA) of nodes 5 and 1 is 3. Another example is LCA of nodes 5 and 4 is 5, since a node can be a descendant of itself according to the LCA definition.
```




方法一：递归
这种方法非常直观。先深度遍历改树。当你遇到节点 p 或 q 时，返回一些布尔标记。该标志有助于确定是否在任何路径中找到了所需的节点。最不常见的祖先将是两个子树递归都返回真标志的节点。它也可以是一个节点，它本身是p或q中的一个，对于这个节点,子树递归返回一个真标志。

让我们看看基于这个想法的形式算法。

算法：

从根节点开始遍历树。
如果当前节点本身是 p 或 q 中的一个，我们会将变量 mid 标记为 true，并继续搜索左右分支中的另一个节点。
如果左分支或右分支中的任何一个返回 true，则表示在下面找到了两个节点中的一个。
如果在遍历的任何点上，左、右或中三个标志中的任意两个变为 true，这意味着我们找到了节点 p 和 q 的最近公共祖先。
让我们看一个示例，然后搜索树中两个节点 9 和 11 的最近公共祖先。

![img](https://pic.leetcode-cn.com/Figures/236/236_LCA_Binary_4.png)
以下是递归中遵循的节点序列：

1 --> 2 --> 4 --> 8
回溯 8 --> 4
4 --> 9 (找到一个节点，返回true)
回溯 9 --> 4 --> 2
2 --> 5 --> 10
回溯 10 --> 5
5 --> 11 (找到另一个节点，返回true)
回溯 --> 5 --> 2

在2节点这里我们有左边=true，右边=true，因此 2 节点是它们的最近公共祖先
JavaPython
class Solution {

    private TreeNode ans;
    
    public Solution() {
        // Variable to store LCA node.
        this.ans = null;
    }
    
    private boolean recurseTree(TreeNode currentNode, TreeNode p, TreeNode q) {
    
        // If reached the end of a branch, return false.
        if (currentNode == null) {
            return false;
        }
    
        // Left Recursion. If left recursion returns true, set left = 1 else 0
        int left = this.recurseTree(currentNode.left, p, q) ? 1 : 0;
    
        // Right Recursion
        int right = this.recurseTree(currentNode.right, p, q) ? 1 : 0;
    
        // If the current node is one of p or q
        int mid = (currentNode == p || currentNode == q) ? 1 : 0;


        // If any two of the flags left, right or mid become True
        if (mid + left + right >= 2) {
            this.ans = currentNode;
        }
    
        // Return true if any one of the three bool values is True.
        return (mid + left + right > 0);
    }
    
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        // Traverse the tree
        this.recurseTree(root, p, q);
        return this.ans;
    	}
    }
## 6. 从有序数组中构造二叉查找树

[108. Convert Sorted Array to Binary Search Tree (Easy)](https://leetcode.com/problems/convert-sorted-array-to-binary-search-tree/description/)

```
public TreeNode sortedArrayToBST(int[] nums) {
    return toBST(nums, 0, nums.length - 1);
}

private TreeNode toBST(int[] nums, int sIdx, int eIdx){
    if (sIdx > eIdx) return null;
    int mIdx = (sIdx + eIdx) / 2;
    TreeNode root = new TreeNode(nums[mIdx]);
    root.left =  toBST(nums, sIdx, mIdx - 1);
    root.right = toBST(nums, mIdx + 1, eIdx);
    return root;
}
```

## 7. 根据有序链表构造平衡的二叉查找树

[109. Convert Sorted List to Binary Search Tree (Medium)](https://leetcode.com/problems/convert-sorted-list-to-binary-search-tree/description/)

```
Given the sorted linked list: [-10,-3,0,5,9],

One possible answer is: [0,-3,9,-10,null,5], which represents the following height balanced BST:

      0
     / \
   -3   9
   /   /
 -10  5
```

方法 1：递归
想法

题目中最重要的要求是需要利用链表中的节点，构建一颗高度平衡的二叉搜索树，好消息是链表中的元素是升序的。

众所周知，一棵二叉搜索树是一棵有根二叉树并且对于所有节点满足特殊的性质：对于树中任意一个点，它的权值必然 \geq≥ 所有左子树节点的权值，\leq≤ 所有右子树节点的权值。因为二叉树具有递归的子结构，二叉搜索树也同理：所有子树也是二叉搜索树。

当前方法和下一个方法的主要思路是：

给定列表中的中间元素将会作为二叉搜索树的根，该点左侧的所有元素递归的去构造左子树，同理右侧的元素构造右子树。这必然能够保证最后构造出的二叉搜索树是平衡的。

算法

由于我们得到的是一个有序链表而不是数组，我们不能直接使用下标来访问元素。我们需要知道链表中的中间元素。
我们可以利用两个指针来访问链表中的中间元素。假设我们有两个指针 slow_ptr 和 fast_ptr。slow_ptr 每次向后移动一个节点而 fast_ptr 每次移动两个节点。当 fast_ptr 到链表的末尾时 slow_ptr 就访问到链表的中间元素。对于一个偶数长度的数组，中间两个元素都可用来作二叉搜索树的根。
当找到链表中的中间元素后，我们将链表从中间元素的左侧断开，做法是使用一个 prev_ptr 的指针记录 slow_ptr 之前的元素，也就是满足 prev_ptr.next = slow_ptr。断开左侧部分就是让 prev_ptr.next = None。
我们只需要将链表的头指针传递给转换函数，进行高度平衡二叉搜索树的转换。所以递归调用的时候，左半部分我们传递原始的头指针；右半部分传递 slow_ptr.next 作为头指针。
通过样例来理解算法的具体过程。

    public TreeNode sortedListToBST(ListNode head) {
        if (head == null) return null;
        if (head.next == null) return new TreeNode(head.val);
        ListNode preMid = preMid(head);
        ListNode mid = preMid.next;
        preMid.next = null;  // 断开链表
        TreeNode t = new TreeNode(mid.val);
        t.left = sortedListToBST(head);
        t.right = sortedListToBST(mid.next);
        return t;
    }
    
    private ListNode preMid(ListNode head) {
        ListNode slow = head, fast = head.next;
        ListNode pre = head;
        while (fast != null && fast.next != null) {
            pre = slow;
            slow = slow.next;
            fast = fast.next.next;
        }
        return pre;
    }



方法 2：递归 + 转成数组
这个方法是空间换时间的经典案例。

你可以通过使用更多空间来降低时间复杂度。

在这个方法中，我们将给定的链表转成数组并利用数组来构建二叉搜索树。数组找中间元素只需要 O(1)O(1) 的时间，所以会降低整个算法的时间复杂度开销。

算法

将给定链表转成数组，将数组的头和尾记成 left 和 right 。
找到中间元素 (left + right) / 2，记为 mid。这需要 O(1)O(1) 时间开销，也是与上面算法主要改进的地方。
将中间元素作为二叉搜索树的根。
递归构造二叉搜索树的左右两棵子树，两个子数组分别是 (left, mid - 1) 和 (mid + 1, right)。
下面是算法的实现部分以及复杂度分析。



    /**
    
    - Definition for singly-linked list. public class ListNode { int val; ListNode next; ListNode(int
    - x) { val = x; } }
      */
      /**
    - Definition for a binary tree node. public class TreeNode { int val; TreeNode left; TreeNode
    - right; TreeNode(int x) { val = x; } }
      */
      class Solution {
    
      private List<Integer> values;
    
      public Solution() {
        this.values = new ArrayList<Integer>();
      }
    
      private void mapListToValues(ListNode head) {
        while (head != null) {
          this.values.add(head.val);
          head = head.next;
        }
      }
    
      private TreeNode convertListToBST(int left, int right) {
        // Invalid case
        if (left > right) {
          return null;
        }
    // Middle element forms the root.
    int mid = (left + right) / 2;
    TreeNode node = new TreeNode(this.values.get(mid));
    
    // Base case for when there is only one element left in the array
    if (left == right) {
      return node;
    }
    
    // Recursively form BST on the two halves
    node.left = convertListToBST(left, mid - 1);
    node.right = convertListToBST(mid + 1, right);
    return node;
      }
    
      public TreeNode sortedListToBST(ListNode head) {
    // Form an array out of the given linked list and then
    // use the array to form the BST.
    this.mapListToValues(head);
    
    // Convert the array to
    return convertListToBST(0, this.values.size() - 1);
      }
    }

复杂度分析

时间复杂度：时间复杂度降到了 O(N)O(N) ，因为需要将链表转成数组。而取中间元素的开销变成了 O(1)O(1) 所以整体的时间复杂度降低了。
空间复杂度：因为我们利用额外空间换取了时间复杂度的降低，空间复杂度变成了 O(N)O(N)，相较于之前算法的 O(\log N)O(logN) 有所提升，因为创建数组的开销。
方法 3：中序遍历模拟
想法

我们知道，二叉树有三种不同的遍历方法：

前序遍历
中序遍历 和
后序遍历。
中序遍历一棵二叉搜索树会有一个非常有趣的结论。

中序遍历一棵二叉搜索树的结果是得到一个升序序列。

这个方法模拟了二叉搜索树的构造过程，因为我们已经获得有序的链表，所以自然的产生了这样的想法。

在描述算法之前，先看一下中序遍历是如何获得有序值的。

基于解决这个问题的中序遍历的思想：

我们知道中序遍历最左边的元素一定是给定链表的头部，类似地下一个元素一定是链表的下一个元素，以此类推。这是肯定的因为给定的初始链表保证了升序排列。

在了解了中序遍历二叉搜索树和有序数组的关系之后，让我们来看看算法。

算法

首先用伪代码来理解一下算法。

➔ function formBst(start, end)
➔      mid = (start + end) / 2
➔      formBst(start, mid - 1)
➔
➔      TreeNode(head.val)
➔      head = head.next
➔       
➔      formBst(mid + 1, end)
➔
遍历整个链表获得它的长度，我们用两个指针标记结果数组的开始和结束，记为 start 和 end，他们的初始值分别为 0 和 length - 1。
记住，我们当前需要模拟中序遍历，找到中间元素 (start + end) / 2。注意这里并不需要在链表中找到确定的元素是哪个，只需要用一个变量告诉我们中间元素的下标。我们只需要递归调用这两侧。
递归左半边，其中开始和结束的值分别为 start, mid - 1。
在这个算法中，每当我们构建完二叉搜索树的左半部分时，链表中的头指针将指向根节点或中间节点（它成为根节点）。 因此，我们只需使用头指针指向的当前值作为根节点，并将指针后移一位，即 head = head.next。
我们在递归右半部分 mid + 1, end。
看一下算法的模拟，理解的更清楚一点。



    JavaPython
    /**
    
    - Definition for singly-linked list. public class ListNode { int val; ListNode next; ListNode(int
    - x) { val = x; } }
      */
      /**
    - Definition for a binary tree node. public class TreeNode { int val; TreeNode left; TreeNode
    - right; TreeNode(int x) { val = x; } }
      */
      class Solution {
    
      private ListNode head;
    
      private int findSize(ListNode head) {
        ListNode ptr = head;
        int c = 0;
        while (ptr != null) {
          ptr = ptr.next;  
          c += 1;
        }
        return c;
      }
    
      private TreeNode convertListToBST(int l, int r) {
        // Invalid case
        if (l > r) {
          return null;
        }
    int mid = (l + r) / 2;
    
    // First step of simulated inorder traversal. Recursively form
    // the left half
    TreeNode left = this.convertListToBST(l, mid - 1);
    
    // Once left half is traversed, process the current node
    TreeNode node = new TreeNode(this.head.val);
    node.left = left;
    
    // Maintain the invariance mentioned in the algorithm
    this.head = this.head.next;
    
    // Recurse on the right hand side and form BST out of them
    node.right = this.convertListToBST(mid + 1, r);
    return node;
      }
    
      public TreeNode sortedListToBST(ListNode head) {
        // Get the size of the linked list first
        int size = this.findSize(head);
    this.head = head;
    // Form the BST now that we know the size
    return convertListToBST(0, size - 1);


​    
​      }
​    }

复杂度分析

时间复杂度：时间复杂度仍然为 O(N)O(N) 因为我们需要遍历链表中所有的顶点一次并构造相应的二叉搜索树节点。
空间复杂度：O(\log N)O(logN) ，额外空间只有一个递归栈，由于是一棵高度平衡的二叉搜索树，所以高度上界为 \log NlogN。

## 10. 寻找二叉查找树中出现次数最多的值

[501. Find Mode in Binary Search Tree (Easy)](https://leetcode.com/problems/find-mode-in-binary-search-tree/description/)

```
   1
    \
     2
    /
   2

return [2].
```

答案可能不止一个，也就是有多个值出现的次数一样多。

```
private int curCnt = 1;
private int maxCnt = 1;
private TreeNode preNode = null;

public int[] findMode(TreeNode root) {
    List<Integer> maxCntNums = new ArrayList<>();
    inOrder(root, maxCntNums);
    int[] ret = new int[maxCntNums.size()];
    int idx = 0;
    for (int num : maxCntNums) {
        ret[idx++] = num;
    }
    return ret;
}

private void inOrder(TreeNode node, List<Integer> nums) {
    if (node == null) return;
    inOrder(node.left, nums);
    if (preNode != null) {
        if (preNode.val == node.val) curCnt++;
        else curCnt = 1;
    }
    if (curCnt > maxCnt) {
        maxCnt = curCnt;
        nums.clear();
        nums.add(node.val);
    } else if (curCnt == maxCnt) {
        nums.add(node.val);
    }
    preNode = node;
    inOrder(node.right, nums);
}
```

# Trie

[![img](https://github.com/CyC2018/CS-Notes/raw/master/notes/pics/5c638d59-d4ae-4ba4-ad44-80bdc30f38dd.jpg)](https://github.com/CyC2018/CS-Notes/blob/master/notes/pics/5c638d59-d4ae-4ba4-ad44-80bdc30f38dd.jpg)

Trie，又称前缀树或字典树，用于判断字符串是否存在或者是否具有某种字符串前缀。

## 1. 实现一个 Trie

[208. Implement Trie (Prefix Tree) (Medium)](https://leetcode.com/problems/implement-trie-prefix-tree/description/)

```
class Trie {

    private class Node {
        Node[] childs = new Node[26];
        boolean isLeaf;
    }

    private Node root = new Node();

    public Trie() {
    }

    public void insert(String word) {
        insert(word, root);
    }

    private void insert(String word, Node node) {
        if (node == null) return;
        if (word.length() == 0) {
            node.isLeaf = true;
            return;
        }
        int index = indexForChar(word.charAt(0));
        if (node.childs[index] == null) {
            node.childs[index] = new Node();
        }
        insert(word.substring(1), node.childs[index]);
    }

    public boolean search(String word) {
        return search(word, root);
    }

    private boolean search(String word, Node node) {
        if (node == null) return false;
        if (word.length() == 0) return node.isLeaf;
        int index = indexForChar(word.charAt(0));
        return search(word.substring(1), node.childs[index]);
    }

    public boolean startsWith(String prefix) {
        return startWith(prefix, root);
    }

    private boolean startWith(String prefix, Node node) {
        if (node == null) return false;
        if (prefix.length() == 0) return true;
        int index = indexForChar(prefix.charAt(0));
        return startWith(prefix.substring(1), node.childs[index]);
    }

    private int indexForChar(char c) {
        return c - 'a';
    }
}
```

# 1. 用栈实现队列

[232. Implement Queue using Stacks (Easy)](https://leetcode.com/problems/implement-queue-using-stacks/description/)

栈的顺序为后进先出，而队列的顺序为先进先出。使用两个栈实现队列，一个元素需要经过两个栈才能出队列，在经过第一个栈时元素顺序被反转，经过第二个栈时再次被反转，此时就是先进先出顺序。

```
class MyQueue {

    private Stack<Integer> in = new Stack<>();
    private Stack<Integer> out = new Stack<>();

    public void push(int x) {
        in.push(x);
    }

    public int pop() {
        in2out();
        return out.pop();
    }

    public int peek() {
        in2out();
        return out.peek();
    }

    private void in2out() {
        if (out.isEmpty()) {
            while (!in.isEmpty()) {
                out.push(in.pop());
            }
        }
    }

    public boolean empty() {
        return in.isEmpty() && out.isEmpty();
    }
}
```

# 2. 用队列实现栈

[225. Implement Stack using Queues (Easy)](https://leetcode.com/problems/implement-stack-using-queues/description/)

在将一个元素 x 插入队列时，为了维护原来的后进先出顺序，需要让 x 插入队列首部。而队列的默认插入顺序是队列尾部，因此在将 x 插入队列尾部之后，需要让除了 x 之外的所有元素出队列，再入队列。

```
class MyStack {

    private Queue<Integer> queue;

    public MyStack() {
        queue = new LinkedList<>();
    }

    public void push(int x) {
        queue.add(x);
        int cnt = queue.size();
        while (cnt-- > 1) {
            queue.add(queue.poll());
        }
    }

    public int pop() {
        return queue.remove();
    }

    public int top() {
        return queue.peek();
    }

    public boolean empty() {
        return queue.isEmpty();
    }
}
```

# 3. 最小值栈

[155. Min Stack (Easy)](https://leetcode.com/problems/min-stack/description/)

```
class MinStack {

    private Stack<Integer> dataStack;
    private Stack<Integer> minStack;
    private int min;

    public MinStack() {
        dataStack = new Stack<>();
        minStack = new Stack<>();
        min = Integer.MAX_VALUE;
    }

    public void push(int x) {
        dataStack.add(x);
        min = Math.min(min, x);
        minStack.add(min);
    }

    public void pop() {
        dataStack.pop();
        minStack.pop();
        min = minStack.isEmpty() ? Integer.MAX_VALUE : minStack.peek();
    }

    public int top() {
        return dataStack.peek();
    }

    public int getMin() {
        return minStack.peek();
    }
}
```

对于实现最小值队列问题，可以先将队列使用栈来实现，然后就将问题转换为最小值栈，这个问题出现在 编程之美：3.7。

# 4. 用栈实现括号匹配

[20. Valid Parentheses (Easy)](https://leetcode.com/problems/valid-parentheses/description/)

```
"()[]{}"

Output : true
public boolean isValid(String s) {
    Stack<Character> stack = new Stack<>();
    for (char c : s.toCharArray()) {
        if (c == '(' || c == '{' || c == '[') {
            stack.push(c);
        } else {
            if (stack.isEmpty()) {
                return false;
            }
            char cStack = stack.pop();
            boolean b1 = c == ')' && cStack != '(';
            boolean b2 = c == ']' && cStack != '[';
            boolean b3 = c == '}' && cStack != '{';
            if (b1 || b2 || b3) {
                return false;
            }
        }
    }
    return stack.isEmpty();
}
```

# 5. 数组中元素与下一个比它大的元素之间的距离

[739. Daily Temperatures (Medium)](https://leetcode.com/problems/daily-temperatures/description/)

```
Input: [73, 74, 75, 71, 69, 72, 76, 73]
Output: [1, 1, 4, 2, 1, 1, 0, 0]
```

在遍历数组时用栈把数组中的数存起来，如果当前遍历的数比栈顶元素来的大，说明栈顶元素的下一个比它大的数就是当前元素。

```
public int[] dailyTemperatures(int[] temperatures) {
    int n = temperatures.length;
    int[] dist = new int[n];
    Stack<Integer> indexs = new Stack<>();
    for (int curIndex = 0; curIndex < n; curIndex++) {
        while (!indexs.isEmpty() && temperatures[curIndex] > temperatures[indexs.peek()]) {
            int preIndex = indexs.pop();
            dist[preIndex] = curIndex - preIndex;
        }
        indexs.add(curIndex);
    }
    return dist;
}
```

```java
/**
 * 根据题意，从最后一天推到第一天，这样会简单很多。因为最后一天显然不会再有升高的可能，结果直接为0。
 * 再看倒数第二天的温度，如果比倒数第一天低，那么答案显然为1，如果比倒数第一天高，又因为倒数第一天
 * 对应的结果为0，即表示之后不会再升高，所以倒数第二天的结果也应该为0。
 * 自此我们容易观察出规律，要求出第i天对应的结果，只需要知道第i+1天对应的结果就可以：
 * - 若T[i] < T[i+1]，那么res[i]=1；
 * - 若T[i] > T[i+1]
 *   - res[i+1]=0，那么res[i]=0;
 *   - res[i+1]!=0，那就比较T[i]和T[i+1+res[i+1]]（即将第i天的温度与比第i+1天大的那天的温度进行比较）
 */
public int[] dailyTemperatures(int[] T) {
    int[] res = new int[T.length];
    res[T.length - 1] = 0;
    for (int i = T.length - 2; i >= 0; i--) {
        for (int j = i + 1; j < T.length; j += res[j]) {
            if (T[i] < T[j]) {
                res[i] = j - i;
                break;
            } else if (res[j] == 0) {
                res[i] = 0;
                break;
            }
        }
    }
    return res;
}
```

# 6. 循环数组中比当前元素大的下一个元素

[503. Next Greater Element II (Medium)](https://leetcode.com/problems/next-greater-element-ii/description/)

```
Input: [1,2,1]
Output: [2,-1,2]
Explanation: The first 1's next greater number is 2;
The number 2 can't find next greater number;
The second 1's next greater number needs to search circularly, which is also 2.
```

与 739. Daily Temperatures (Medium) 不同的是，数组是循环数组，并且最后要求的不是距离而是下一个元素。

```
public int[] nextGreaterElements(int[] nums) {
    int n = nums.length;
    int[] next = new int[n];
    Arrays.fill(next, -1);
    Stack<Integer> pre = new Stack<>();
    for (int i = 0; i < n * 2; i++) {
        int num = nums[i % n];
        while (!pre.isEmpty() && nums[pre.peek()] < num) {
            next[pre.pop()] = num;
        }
        if (i < n){
            pre.push(i);
        }
    }
    return next;
}
```



```java
class Solution {
    public int[] nextGreaterElements(int[] nums) {
 		int length= nums.length;
 		int[] result= new int[length];
 		for(int i=0;i<length;i++){
 			int j=i+1;
 			for (;j%length!=i;j++ ) {
 				if(nums[j%length]>nums[i]){
 					result[i]=nums[j%length];
 					break;
 				}
 			}
 			if(j%length==i)
 				result[i]=-1;
 		}
 		return result;       
    }
}
```

# 2. 判断数组是否含有重复元素

[217. Contains Duplicate (Easy)](https://leetcode.com/problems/contains-duplicate/description/)

方法三：哈希表 【通过】
直觉

利用支持快速搜索和插入操作的动态数据结构。

算法

从方法一中我们知道，对无序数组的查找操作的时间复杂度为 O(n)O(n)，而我们会重复调用查找操作。因此，使用搜索时间更快的数据结构将加快整个算法的速度。

有许多数据结构常用作动态集合,如二进制搜索树和哈希表。这里我们需要的操作是 search 和 insert。对于平衡二叉搜索树（Java 中的 TreeSet 或 TreeMap），search 和 insert 的时间复杂度均为 O(\log n)O(logn)。对于哈希表（Java 中的 HashSet 或 HashMap），search 和 insert 的平均时间复杂度为 O(1)O(1)。因此，通过使用哈希表，我们可以达到在线性时间复杂度解决问题。

```java
public boolean containsDuplicate(int[] nums) {
    Set<Integer> set = new HashSet<>(nums.length);
    for (int x: nums) {
        if (set.contains(x)) return true;
        set.add(x);
    }
    return false;
}
```



# 3. 最长和谐序列

[594. Longest Harmonious Subsequence (Easy)](https://leetcode.com/problems/longest-harmonious-subsequence/description/)

```
Input: [1,3,2,2,5,2,3,7]
Output: 5
Explanation: The longest harmonious subsequence is [3,2,2,2,3].
```

和谐序列中最大数和最小数之差正好为 1，应该注意的是序列的元素不一定是数组的连续元素。

```
public int findLHS(int[] nums) {
    Map<Integer, Integer> countForNum = new HashMap<>();
    for (int num : nums) {
        countForNum.put(num, countForNum.getOrDefault(num, 0) + 1);
    }
    int longest = 0;
    for (int num : countForNum.keySet()) {
        if (countForNum.containsKey(num + 1)) {
            longest = Math.max(longest, countForNum.get(num + 1) + countForNum.get(num));
        }
    }
    return longest;
}
```

# 4. 最长连续序列

[128. Longest Consecutive Sequence (Hard)](https://leetcode.com/problems/longest-consecutive-sequence/description/)

```
Given [100, 4, 200, 1, 3, 2],
The longest consecutive elements sequence is [1, 2, 3, 4]. Return its length: 4.
```

方法 2：排序
想法

如果我们可以将数组中的数字升序迭代，找连续数字会变得十分容易。为了将数组变得有序，我们将数组进行排序。

算法

在我们开始算法之前，首先检查输入的数组是否为空数组，如果是，函数直接返回 0 。对于其他情况，我们将 nums 数组排序，并考虑除了第一个数字以外的每个数字与它前一个数字的关系。如果当前数字和前一个数字相等，那么我们当前的序列既不会增长也不会中断，我们只需要继续考虑下一个数字。如果不相等，我们必须要检查当前数字是否能延长答案序列（也就是 nums[i] == nums[i-1] + 1）。如果可以增长，那么我们将当前数字添加到当前序列并继续。否则，当前序列被中断，我们记录当前序列的长度并将序列长度重置为 1 。由于 nums 中的最后一个数字也可能是答案序列的一部分，所以我们将当前序列的长度和记录下来的最长序列的长度的较大值返回。



```java
class Solution {
   public int longestConsecutive(int[] nums) {
        if (nums.length == 0) {
            return 0;
        }
    Arrays.sort(nums);

    int longestStreak = 1;
    int currentStreak = 1;

    for (int i = 1; i < nums.length; i++) {
        if (nums[i] != nums[i-1]) {
            if (nums[i] == nums[i-1]+1) {
                currentStreak += 1;
            }
            else {
                longestStreak = Math.max(longestStreak, currentStreak);
                currentStreak = 1;
            }
        }
    }

    return Math.max(longestStreak, currentStreak);
	}
}


```
方法 3：哈希表和线性空间的构造
想法

其实我们一开始的暴力解法的思路是正确的，但是需要进行一些优化，才能达到 O(n)O(n) 的时间复杂度。

算法

这个优化算法与暴力算法仅有两处不同：这些数字用一个 HashSet 保存（或者用 Python 里的 Set），实现 O(1)O(1) 时间的查询，同时，我们只对 当前数字 - 1 不在哈希表里的数字，作为连续序列的第一个数字去找对应的最长序列，这是因为其他数字一定已经出现在了某个序列里。

```java
class Solution {
    public int longestConsecutive(int[] nums) {
        Set<Integer> num_set = new HashSet<Integer>();
        for (int num : nums) {
            num_set.add(num);
        }

        int longestStreak = 0;

        for (int num : num_set) {
            if (!num_set.contains(num-1)) {
                int currentNum = num;
                int currentStreak = 1;

                while (num_set.contains(currentNum+1)) {
                    currentNum += 1;
                    currentStreak += 1;
                }

                longestStreak = Math.max(longestStreak, currentStreak);
            }
        }

        return longestStreak;
    }
}

```

