// ------------------------------------------------------------------------
//  Copyright (C)
//  Jordi Pont-Tuset <jponttuset@vision.ee.ethz.ch>
//  July 2015
// ------------------------------------------------------------------------

#ifndef MOVE2JUNCTIONS_HPP
#define MOVE2JUNCTIONS_HPP

#include <vector>
#include <list>

#include <cpp/containers.hpp>

namespace PolyCont
{
    Point cont_next_extreme(const Point& cont_coord, const Point& last_coord)
    {
        Point to_return = last_coord;
    // mexPrintf("Next from (%d,%d) to (%d,%d):", (int)last_coord.X, (int)last_coord.Y, (int)cont_coord.X, (int)cont_coord.Y);  
        if(cont_coord.X%2==0) /* Horizontal contour */
        {
            if (cont_coord.Y+1==last_coord.Y)
                to_return.Y -= 2;
            else if (cont_coord.Y-1==last_coord.Y)
                to_return.Y += 2;
            else
                assert(0);
                // mexErrMsgTxt("cont_next_extreme 1: Something went wrong");
        }
        else if(cont_coord.Y%2==0) /* Vertical contour */
        {
            if (cont_coord.X+1==last_coord.X)
                to_return.X -=2;
            else if (cont_coord.X-1==last_coord.X)
                to_return.X +=2;
            else
                assert(0);
                // mexErrMsgTxt("cont_next_extreme 2: Something went wrong");
        }
        else
            assert(0);
            // mexErrMsgTxt("cont_next_extreme 3: Something went wrong");

    // mexPrintf(" (%d,%d)\n", (int)to_return.X, (int)to_return.Y);  

        return to_return;
    }


    void move2junctions(Path& contour, const Junction& junct)
    {
        Point next_to_explore = contour[0];
        contour[0] = junct.pos;
        for(std::size_t jj=1; jj<contour.size(); ++jj)
        {
            Point next_pos = cont_next_extreme(next_to_explore, contour[jj-1]);
            next_to_explore = contour[jj];
            contour[jj] = next_pos;
        }

        Point next_pos = cont_next_extreme(next_to_explore, contour[contour.size()-1]);
        assert(next_pos==junct.pos);
        //    mexErrMsgTxt("Something went wrong: next_pos!=junct.pos");

        contour.push_back(junct.pos);
    }
}

#endif
