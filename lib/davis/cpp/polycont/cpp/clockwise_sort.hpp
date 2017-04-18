// ------------------------------------------------------------------------
//  Copyright (C)
//  Jordi Pont-Tuset <jponttuset@vision.ee.ethz.ch>
//  July 2015
// ------------------------------------------------------------------------

#ifndef CLOCKWISESORT_HPP
#define CLOCKWISESORT_HPP

#include <vector>

#include <cpp/containers.hpp>

namespace PolyCont
{
    bool is_clockwise(const Path& polig)
    {
        float total = 0.0;
        for(std::size_t ii=0; ii<polig.size(); ++ii)
        {
            std::size_t jj = (ii + 1) % polig.size();
            total += (polig[jj].X-polig[ii].X)*(polig[jj].Y+polig[ii].Y);
        }

        return total > 0;
    }


    void clockwise_sort(ContContainer& all_conts)
    {
        /* Sort clockwise the outside ones and counterclockwise the holes */
        for(std::size_t ii=0; ii<all_conts.size(); ++ii)
        {
            if (!all_conts.is_hole[ii])
            {
                if (not is_clockwise(all_conts[ii]))
                    std::reverse(all_conts[ii].begin(),all_conts[ii].end());
            }
            else
            {
                if (is_clockwise(all_conts[ii]))
                    std::reverse(all_conts[ii].begin(),all_conts[ii].end());
            }
        }
    }
}
#endif
