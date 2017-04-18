// ------------------------------------------------------------------------
//  Copyright (C)
//  Jordi Pont-Tuset <jponttuset@vision.ee.ethz.ch>
//  July 2015
// ------------------------------------------------------------------------

#ifndef INOUTSORT_HPP
#define INOUTSORT_HPP

#include <set>
#include <map>
#include <string>
#include <deque>
#include <iostream>
#include <vector>
#include <iomanip>


#include <cpp/containers.hpp>
    
namespace PolyCont
{
    
    /* Container of the contour inclusion relations (holes) */
    struct cont_inc
    {
        std::vector<int> all_parents;
        std::vector<std::set<std::size_t>> all_children;
    };
    
    void flatten_hierarchy(std::size_t depth, std::size_t curr_parent, cont_inc& inc)
    {
        for(auto child:inc.all_children[curr_parent])
        {
            if (depth%2)
                inc.all_parents[child]  = -1;
            else
                inc.all_parents[child] -= 1;
            flatten_hierarchy(depth+1, child, inc);
        }
    }

    bool rayintersectseg(const Eigen::Vector2d& point, const Eigen::Vector2d& seg1, const Eigen::Vector2d& seg2)
    {
        Eigen::Vector2d p = point;
        Eigen::Vector2d a = seg1;
        Eigen::Vector2d b = seg2;

        if (a(1) > b(1))
        {
            Eigen::Vector2d tmp = a;
            a = b;
            b = tmp;
        }
        if (p(1) == a(1) or p(1) == b(1))
            p(1) += std::numeric_limits<float>::epsilon();


        if ((p(1) > b(1) or p(1) < a(1)) or (p(0) > std::max(a(0), b(0))))
            return false;

        if (p(0) < std::min(a(0), b(0)))
            return true;
        else
        {
            float m_red;
            float m_blue;
            if (fabs(a(0) - b(0)) > std::numeric_limits<float>::min())
                m_red = (b(1) - a(1)) / (float)(b(0) - a(0));
            else
                m_red = std::numeric_limits<float>::max();
            if (fabs(a(0) - p(0)) > std::numeric_limits<float>::min())
                m_blue = (p(1) - a(1)) / (float)(p(0) - a(0));
            else
                m_blue = std::numeric_limits<float>::max();
            return m_blue >= m_red;
        }
    }

    bool is_point_inside(const Eigen::Vector2d& p, const MatrixType& poly)
    {
        std::size_t cross_times = 0;
        for(std::size_t seg_id=0; seg_id<poly.rows()-1; ++seg_id)
            cross_times += rayintersectseg(p, poly.row(seg_id), poly.row(seg_id+1));
        cross_times += rayintersectseg(p, poly.row(0), poly.row(poly.rows()-1));

        return cross_times % 2 == 1;
    }


    void add_one_inout(std::size_t curr_cont, std::size_t curr_parent, const std::vector<MatrixType>& all_conts, cont_inc& inc)
    {
        /* Check whether it's inside a children */
        int found_child = -1;
        for(auto child : inc.all_children[curr_parent])
        {
            if(is_point_inside(all_conts[curr_cont-1].row(0), all_conts[child-1]))
            {
                found_child = (int)child;
                break;
            }
        }

        if (found_child>=0)
        {
            /* If yes --> Reiterate with parent = children */
            add_one_inout(curr_cont, (std::size_t)found_child, all_conts, inc);
        }
        else
        {
            /* If not, check which children are in it */
            std::set<std::size_t> children_in_it;
            for(auto child : inc.all_children[curr_parent])
                if(is_point_inside(all_conts[child-1].row(0), all_conts[curr_cont-1]))
                    children_in_it.insert(child);

            if(not children_in_it.empty())
            {
                /* If there are children, then create another child,
                 and the old children as a children of the new one */
                inc.all_children.push_back(children_in_it);
                for(auto child : children_in_it)
                {
                    inc.all_children[curr_parent].erase(child);
                    inc.all_parents[child] = (int)curr_cont;
                }
            }
            else
            {
                /* If not, then create a new child */
                inc.all_children.push_back(std::set<std::size_t>());
            }

            /* Add the child to the parent */
            inc.all_children[curr_parent].insert(curr_cont);
            inc.all_parents.push_back((int)curr_parent);
        }
    }



    std::vector<bool> in_out_check(const std::vector<MatrixType>& all_conts)
    {
        cont_inc inc;

        /* Empty case */
        if (all_conts.empty())
            return std::vector<bool>();

        /* Add root with empty children */
        inc.all_parents.push_back(-1);
        inc.all_children.push_back(std::set<std::size_t>());

        /* Add all pieces of contours */
        for(std::size_t cont_id=0; cont_id<all_conts.size(); ++cont_id)
            add_one_inout(cont_id+1, 0, all_conts, inc);

        /* Flatten the hierarchy (just pairs of inclusions) */
        flatten_hierarchy(1, 0, inc);

        /* Remove the root of the tree */
        inc.all_parents.erase(inc.all_parents.begin());
        inc.all_children.erase(inc.all_children.begin());

//         /* Remove children from the non roots */
//         for(std::size_t ii=0; ii<inc.all_parents.size(); ++ii)
//         {
//             if (inc.all_parents[ii]>=0)
//                 inc.all_children[ii].clear();
//             else
//             {
//                 /* -1 to all */
//                 std::set<std::size_t> new_children;
//                 for(auto curr_child:inc.all_children[ii])
//                     new_children.insert(curr_child-1);
//                 inc.all_children[ii] = new_children;
//             }
//         }
//         return inc;
        
        /* Store if they are holes */
        std::vector<bool> is_hole(inc.all_parents.size());
        for(std::size_t ii=0; ii<inc.all_parents.size(); ++ii)
            is_hole[ii] = (inc.all_parents[ii]!=-1);
        
        return is_hole;
    }

}

#endif
