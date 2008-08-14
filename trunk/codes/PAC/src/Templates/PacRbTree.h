#ifndef PAC_RB_TREE_H
#define PAC_RB_TREE_H

#include "Templates/PacPair.h"
#include "Templates/PacFunction.h"

// #include <tree.h>
// #define PacRbTree rb_tree

#include <bits/stl_tree.h>

using std::_Rb_tree;
using std::_Rb_tree_node;
using std::_Rb_tree_node_base;
using std::_Rb_tree_iterator;
using std::allocator;
using std::pair;

  // Class rb_tree is not part of the C++ standard.  It is provided for
  // compatibility with the HP STL.

  /**
   *  This is an SGI extension.
   *  @ingroup SGIextensions
   *  @doctodo
   */
  template <class _Key, class _Value, class _KeyOfValue, class _Compare,
	    class _Alloc = allocator<_Value> >
    class PacRbTree
    : public _Rb_tree<_Key, _Value, _KeyOfValue, _Compare, _Alloc>
    {

      public:

      typedef _Rb_tree<_Key, _Value, _KeyOfValue, _Compare, _Alloc> _Base;
      typedef typename _Base::allocator_type allocator_type;

      PacRbTree(const _Compare& __comp = _Compare(),
	      const allocator_type& __a = allocator_type())
      : _Base(__comp, __a) { }

      ~PacRbTree() { }

      // typedef _Rb_tree_node<_Value> _Rb_tree_node;
      typedef _Rb_tree_node<_Value>* _Link_type;

      // Insert/erase.

      typedef _Value value_type;

      typedef _Rb_tree_iterator<value_type>       iterator;


      pair<iterator, bool> insert_unique(const value_type& __x);

      iterator M_insert(_Rb_tree_node_base* __x, _Rb_tree_node_base* __y, const value_type& __v);

    };

  template<typename _Key, typename _Val, typename _KeyOfValue,
           typename _Compare, typename _Alloc>
    pair<typename PacRbTree<_Key, _Val, _KeyOfValue, _Compare, _Alloc>::iterator, bool>
    PacRbTree<_Key, _Val, _KeyOfValue, _Compare, _Alloc>::insert_unique(const _Val& __v)
    {
      _Link_type __x = _Base::_M_begin();
      _Link_type __y = _Base::_M_end();

      bool __comp = true;
      while (__x != 0)
	{
	  __y = __x;
	  __comp = _Base::_M_impl._M_key_compare(_KeyOfValue()(__v), _Base::_S_key(__x));
	  __x = __comp ? _Base::_S_left(__x) : _Base::_S_right(__x);
	}
      iterator __j = iterator(__y);
      if (__comp)
	if (__j == _Base::begin())
	  return pair<iterator,bool>(M_insert(__x, __y, __v), true);
	else
	  --__j;
      if (_Base::_M_impl._M_key_compare(_Base::_S_key(__j._M_node), _KeyOfValue()(__v)))
	return pair<iterator, bool>(M_insert(__x, __y, __v), true);
      return pair<iterator, bool>(__j, false);
    }

  template<typename _Key, typename _Val, typename _KeyOfValue,
           typename _Compare, typename _Alloc>
    typename PacRbTree<_Key, _Val, _KeyOfValue, _Compare, _Alloc>::iterator
    PacRbTree<_Key, _Val, _KeyOfValue, _Compare, _Alloc>::
    M_insert(_Rb_tree_node_base* __x, _Rb_tree_node_base* __p, const _Val& __v)
    {
      bool __insert_left = (__x != 0 || __p == _Base::_M_end()
			    || _Base::_M_impl._M_key_compare(_KeyOfValue()(__v), 
							     _Base:: _S_key(__p)));

      _Link_type __z = _Base::_M_create_node(__v);

      _Rb_tree_insert_and_rebalance(__insert_left, __z, __p,  
				    this->_M_impl._M_header);
      ++_Base::_M_impl._M_node_count;
      return iterator(__z);
    }


#endif
