// ------------------------------------------------------------------------
//  Copyright (C)
//  Jordi Pont-Tuset <jponttuset@vision.ee.ethz.ch>
//  July 2015
// ------------------------------------------------------------------------

#ifndef SIMPLIFY_HPP
#define SIMPLIFY_HPP

#include <vector>
#include <list>

#include <cpp/containers.hpp>
#include <psimpl/psimpl.hpp>
#include <Eigen/Dense>

namespace PolyCont
{
    void simplify_one(ContContainer& all_cont_pieces, /* Vector containing the finished contour pieces */
                      double simplify_tol             /* Tolerance of the line simplification algorithm */
                      )
    {
        /* Sweep all contour pieces */
        for (std::size_t ii=0; ii<all_cont_pieces.size(); ++ii)
        {
            /* Copy the data to the structure needed to simplify */
            std::list<double> to_simpl;
            for (std::size_t jj=0; jj<all_cont_pieces[ii].size(); ++jj)
            {
                /* Coordinates in the form x1,y1,x2,y2, etc. */
                to_simpl.push_back(all_cont_pieces[ii][jj].X);
                to_simpl.push_back(all_cont_pieces[ii][jj].Y);
            }
            std::list<double> simplified(to_simpl.size());
            std::fill(simplified.begin(), simplified.end(), -1);

            /* Simplify */
            psimpl::simplify_douglas_peucker<2>(to_simpl.begin(), to_simpl.end(), simplify_tol, simplified.begin());

            /* Copy back to the Coord vector */
            all_cont_pieces[ii].resize(0);
            for(std::list<double>::iterator it=simplified.begin(); it!=simplified.end(); ++it)
            {   
                if(*it==-1)
                    break;
                double tmp_x = *it;
                ++it;
                double tmp_y = *it;

                all_cont_pieces[ii].push_back(Point(tmp_x, tmp_y));
            }
        }
    }


    void simplify(ContContainer& all_cont_pieces, /* Vector containing the finished contour pieces */
                  double simplify_tol             /* Tolerance of the line simplification algorithm */
                  )
    {
        // Do nothing if simplify_tol==0
        if (simplify_tol>0)
        {
            // Do a first simplification of the 'straight' lines
            if (simplify_tol>0.1)
                simplify_one(all_cont_pieces, 0.1);

            // Then do the real simplification
            // (works better in practice, but it can be removed for efficiency)
            simplify_one(all_cont_pieces, simplify_tol);
        }
    }
}
#endif
