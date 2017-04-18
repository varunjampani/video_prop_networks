// ------------------------------------------------------------------------
//  Copyright (C)
//  Jordi Pont-Tuset <jponttuset@vision.ee.ethz.ch>
//  July 2015
// ------------------------------------------------------------------------

#ifndef MASK2POLY_HPP
#define MASK2POLY_HPP

#include <iostream>
#include <list>
#include <set>
#include <map>
#include <algorithm>
#include <cpp/containers.hpp>
#include <Eigen/Dense>

namespace PolyCont
{
    void explore_contour(Path& curr_cont,
                         std::vector<Junction>& junction_set,
                         Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>& visited,
                         Eigen::ArrayXXi& junction_ids,
                         const Eigen::Array<bool,Eigen::Dynamic,Eigen::Dynamic>& is_cont)
    {
        /* This will contain the next contour coordinate to explore at each iteration */
        Point next_to_visit = curr_cont[0];
        curr_cont.pop_back();

        /* Store the next 'Junction point' to be visited */
        Point junct_visited(junction_set[curr_cont.orig_junc-1].pos);
        Point junct_to_visit = junct_visited;

        /* Keep exploring until we find a Junction */
        bool found = false;
        while(!found)
        {
            /* Mark the visited Junction point as visited */
            visited(junct_visited.X+1,junct_visited.Y+1) = true;

            /* Get the coordinate to explore */
            Point cont_to_visit = next_to_visit;

            /* Is it a vertical or horizontal contour? */
            Point extreme1;
            Point extreme2;
            if (cont_to_visit.X % 2) // Vertical
            {
                extreme1.X = cont_to_visit.X+1;
                extreme1.Y = cont_to_visit.Y;
                extreme2.X = cont_to_visit.X-1;
                extreme2.Y = cont_to_visit.Y;
            }
            else // Horizontal
            {
                extreme1.X = cont_to_visit.X;
                extreme1.Y = cont_to_visit.Y-1;
                extreme2.X = cont_to_visit.X;
                extreme2.Y = cont_to_visit.Y+1;
            }

            /* Get the next Junction point to visit */
            if(!visited(extreme1.X+1,extreme1.Y+1))       /* Next 'Junction point' not visited */
            {
                junct_to_visit.X = extreme1.X;
                junct_to_visit.Y = extreme1.Y;
            }
            else if (!visited(extreme2.X+1,extreme2.Y+1)) /* Other not visited */
            {
                junct_to_visit.X = extreme2.X;
                junct_to_visit.Y = extreme2.Y;
            }
            else  /* Both visited, look for the Junction, which will be the end of the piece */
            {
                if (extreme1.X==junct_to_visit.X && extreme1.Y==junct_to_visit.Y)
                {
                    junct_to_visit.X = extreme2.X;
                    junct_to_visit.Y = extreme2.Y;
                }
                else
                {
                    assert(extreme2.X==junct_to_visit.X && extreme2.Y==junct_to_visit.Y);
                        
                    junct_to_visit.X = extreme1.X;
                    junct_to_visit.Y = extreme1.Y;
                }
            }

            /* Mark the contour and Junction as visited */
            visited( cont_to_visit.X+1, cont_to_visit.Y+1) = true;
            visited(junct_to_visit.X+1,junct_to_visit.Y+1) = true;

            /* Is the 'Junction point' to be visited really a Junction? */
            if (junction_ids(junct_to_visit.X,junct_to_visit.Y)>0) /* Yes it is, so we finish */
            {
                found = true;
                curr_cont.end_junc = junction_ids(junct_to_visit.X,junct_to_visit.Y);

    //             std::ostringstream stringStream;
    //             stringStream << "End Junction found (" << junct_to_visit.X << "," << junct_to_visit.Y << ") id:" << curr_cont.end_junc << std::endl;
    //             mexPrintf(stringStream.str().c_str());
            }
            else /* It is not a Junction, keep visiting */
            {
                /* Look for the next unvisited direction with contour*/
                if      (is_cont(junct_to_visit.X+1+1, junct_to_visit.Y  +1) && !visited(junct_to_visit.X+1+1,junct_to_visit.Y  +1))
                    next_to_visit = Point(junct_to_visit.X+1,junct_to_visit.Y);
                else if (is_cont(junct_to_visit.X-1+1, junct_to_visit.Y  +1) && !visited(junct_to_visit.X-1+1,junct_to_visit.Y  +1))
                    next_to_visit = Point(junct_to_visit.X-1,junct_to_visit.Y);
                else if (is_cont(junct_to_visit.X  +1, junct_to_visit.Y+1+1) && !visited(junct_to_visit.X  +1,junct_to_visit.Y+1+1))
                    next_to_visit = Point(junct_to_visit.X,junct_to_visit.Y+1);
                else
                {
                    assert(is_cont(junct_to_visit.X  +1,junct_to_visit.Y-1+1) && !visited(junct_to_visit.X  +1,junct_to_visit.Y-1+1));
//                     { 
//                         std::ostringstream stringStream;
//                         stringStream << "Oh oh: Position (" << junct_to_visit.X << "," << junct_to_visit.Y << ") no unvisited contour found!";
//                         mexErrMsgTxt(stringStream.str().c_str());
//                     }
                    next_to_visit = Point(junct_to_visit.X,junct_to_visit.Y-1);
                }
            }

            /* Add the contour coordinate to the list */
            curr_cont.push_back(Point(cont_to_visit.X,cont_to_visit.Y));
    //         mexPrintf("  Explored (%d,%d)\n",cont_to_visit.X,cont_to_visit.Y);
        }
    }



    /* Implementation of the actual function: from a mask we get the
     * set of junctions and the set of contour pieces that form it.
     */
    template<typename ArrayOrMap>
    void mask2poly(ArrayOrMap & mask,                    /* Input mask */
                   ContContainer& all_cont_pieces,       /* Vector containing the finished contour pieces */
                   Eigen::ArrayXXi& junction_ids,        /* Position of all junctions in the contour grid */
                   std::vector<Junction>& junction_set)  /* Vector containing all Junction descriptions   */
    {
        /* Input size */
        int sx = (int)mask.rows();
        int sy = (int)mask.cols();

        /* Add padding to mask */
        Eigen::Array<bool,Eigen::Dynamic,Eigen::Dynamic> mask_pad = Eigen::Array<bool,Eigen::Dynamic,Eigen::Dynamic>::Zero(sx+2, sy+2);
        for (int xx=1; xx<=sx; ++xx)
            for (int yy=1; yy<=sy; ++yy)
                mask_pad(xx,yy) = mask(xx-1,yy-1);

        /*--------------------------------------------------------------------*/
        /*           Sweep in all directions to get contour places            */
        /*--------------------------------------------------------------------*/
        Eigen::Array<bool,Eigen::Dynamic,Eigen::Dynamic> is_cont = Eigen::Array<bool,Eigen::Dynamic,Eigen::Dynamic>::Zero(2*sx+3,2*sy+3);

        /* Vertical contours */
        for (int xx=0; xx<sx; ++xx)
            for (int yy=-1; yy<sy; ++yy)
                is_cont(2*xx+1+1,2*yy+2+1) = (mask_pad(xx+1,yy+1)^mask_pad(xx+1,yy+1+1));

        /* Horizontal contours */
        for (int xx=-1; xx<sx; ++xx)
            for (int yy=0; yy<sy; ++yy)
                is_cont(2*xx+2+1,2*yy+1+1) = (mask_pad(xx+1,yy+1)^mask_pad(xx+1+1,yy+1));

    //     /* Debug */
    //     mexPrintf("is_cont\n");
    //     for(std::size_t ii=0; ii<is_cont.rows(); ++ii)
    //     {
    //         for(std::size_t jj=0; jj<is_cont.cols(); ++jj)
    //             if (is_cont(ii,jj))
    //                 mexPrintf("1 ");
    //             else
    //                 mexPrintf("0 ");
    //         mexPrintf("\n");
    //     }


        /*-------------------------------------------*/
        /*            Look for junctions             */
        /*-------------------------------------------*/
        /* Containers */
        junction_ids = Eigen::ArrayXXi::Zero(2*sx+1,2*sy+1);
        std::size_t n_junctions = 0;

        /*       1
         *       |
         *  2 -- X -- 4
         *       |
         *       3
         */
        for (std::size_t xx=2; xx<2*sx; xx+=2)
        {
            for (std::size_t yy=2; yy<2*sy; yy+=2)
            {
                Junction tmp_jct = Junction(n_junctions+1,xx,yy);

                std::set<double> curr_neighs;
                if (is_cont(xx  +1,yy+1+1) &
                    is_cont(xx+1+1,yy  +1) & 
                    is_cont(xx  +1,yy-1+1) &
                    is_cont(xx-1+1,yy  +1))
                {
                    tmp_jct.cont_pos.push_back(Point(xx  ,yy+1));
                    tmp_jct.cont_pos.push_back(Point(xx+1,yy  ));
                    tmp_jct.cont_pos.push_back(Point(xx  ,yy-1));
                    tmp_jct.cont_pos.push_back(Point(xx-1,yy  ));
                    junction_set.push_back(tmp_jct);
                    junction_ids(xx,yy) = (int)n_junctions + 1;
                    n_junctions++;
                }
            }
        }

    //     /* Debug */
    //     mexPrintf("junction_ids\n");
    //     for(std::size_t ii=0; ii<junction_ids.rows(); ++ii)
    //     {
    //         for(std::size_t jj=0; jj<junction_ids.cols(); ++jj)
    //             mexPrintf("%d ",junction_ids(ii,jj));
    //         mexPrintf("\n");
    //     }


        /*--------------------------------------------------------------------*/
        /*     Sweep all contour places and connect them in a sorted way      */
        /*--------------------------------------------------------------------*/
        /* Vector containing the next contours to explore */
        std::vector<Path> to_explore;
        /* Mask of the visited positions */
        Eigen::Matrix<bool,Eigen::Dynamic,Eigen::Dynamic> visited = Eigen::Matrix<bool,Eigen::Dynamic,Eigen::Dynamic>::Zero(2*sx+3,2*sy+3);


        /* Explore all junctions to start from there. Please note that the hole contours will *
         * not be added here because they have no Junction, but they will be added afterwards */
        for(std::size_t jj=0; jj<junction_set.size(); ++jj)
            for(std::size_t ii=0; ii<junction_set[jj].cont_pos.size(); ++ii)
                to_explore.push_back(Path(junction_set[jj].id,
                                          junction_set[jj].cont_pos[ii]));

        while (to_explore.size()>0)
        {
            /* Get the current contour to explore */
            Path curr_cont = to_explore.back();
            to_explore.pop_back();

            /* Check it was not explored before */
            if(visited(curr_cont.back().X+1,curr_cont.back().Y+1))
                continue;

    //         mexPrintf("Exploring (%d,%d)\n",curr_cont.back().X,curr_cont.back().Y);
            /* Do the actual exploration */
            explore_contour(curr_cont, junction_set, visited, junction_ids, is_cont);

            /* Store it as finished */
            all_cont_pieces.push_back(curr_cont);

    //         /* Debug */
    //         mexPrintf("visited iteration %d\n",to_explore.size());
    //         for(std::size_t ii=0; ii<visited.rows(); ++ii)
    //         {
    //             for(std::size_t jj=0; jj<visited.cols(); ++jj)
    //                 if (visited(ii,jj))
    //                     mexPrintf("1 ");
    //                 else
    //                     mexPrintf("0 ");
    //             mexPrintf("\n");
    //         }
        }




        /*--------------------------------------------------------------------*/
        /*   Rescan contours to find holes and create artificial junctions    */
        /*--------------------------------------------------------------------*/

        /* Vertical contours */
        /* Horizontal contours are not necessary because there will always be a vertical contour in any hole */
        for (int xx=0; xx<sx; ++xx)
        {
            for (int yy=-1; yy<sy; ++yy)
            {
                if (is_cont(2*xx+1+1,2*yy+2+1) && (visited(2*xx+1+1,2*yy+2+1)==0))
                {
                    /* Create an 'artificial' Junction */
                    n_junctions++;
                    Junction tmp_jct(n_junctions,2*xx,2*yy+2);
                    tmp_jct.cont_pos.push_back(Point(2*xx+1,2*yy+2));

                    /* Add it to the set of junctions */
                    junction_set.push_back(tmp_jct);
                    junction_ids(2*xx,2*yy+2) = (int)n_junctions;


                    /* Create the contour piece to explore */
                    Path curr_cont = Path(tmp_jct.id,
                                                      tmp_jct.cont_pos[0]);

                    /* Do the actual exploration */
                    explore_contour(curr_cont, junction_set, visited, junction_ids, is_cont);

                    /* Store it as finished */
                    all_cont_pieces.push_back(curr_cont);
                }
            }
        }


        /*-------------------------------------------*/
        /*       Fill Junction contour pieces        */
        /*-------------------------------------------*/
        for (std::size_t ii=0; ii<all_cont_pieces.size(); ++ii)
        {
            junction_set[all_cont_pieces[ii].orig_junc-1].cont_pieces.insert(ii);
            junction_set[all_cont_pieces[ii].end_junc-1].cont_pieces.insert(ii);
        }
    }
}

#endif


