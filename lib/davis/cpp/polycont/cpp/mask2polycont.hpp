// ------------------------------------------------------------------------
//  Copyright (C)
//  Jordi Pont-Tuset <jponttuset@vision.ee.ethz.ch>
//  July 2015
// ------------------------------------------------------------------------

#ifndef MASK2POLYCONT_HPP
#define MASK2POLYCONT_HPP

#include <cpp/simplify.hpp>
#include <cpp/mask2poly.hpp>
#include <cpp/in_out_sort.hpp>
#include <cpp/lines2jordan.hpp>
#include <cpp/move2junctions.hpp>
#include <cpp/clockwise_sort.hpp>

namespace PolyCont
{
    void mask2polycont(const MaskType& mask, ContContainer& polycont, double simplify_tol=0)
    {
        /* Store image size */
        /* Antiintuitive because RowMajor storage,
         * but we need it for the map on Matlab storage */
        polycont.im_sx = mask.rows();
        polycont.im_sy = mask.cols();
        
        /*-------------------------------------------*/
        /*         Mask to polygonal line            */
        /*-------------------------------------------*/ 
        ContContainer          all_cont_pieces;
        Eigen::ArrayXXi        junction_ids;
        std::vector<Junction>  junction_set;
        mask2poly(mask, all_cont_pieces, junction_ids, junction_set);
        

        /*-------------------------------------------*/
        /*      Group lines into Jordan curves       */
        /*  (non-self-intersecting continuous loop)  */
        /*-------------------------------------------*/
        lines2jordan(all_cont_pieces, junction_set, polycont);


        /*-------------------------------------------*/
        /*               Sort in-out                 */
        /*-------------------------------------------*/  
        /* Copy data to a Eigen Matrix */
        std::vector<MatrixType> all_conts(polycont.size());
        for (std::size_t ii=0; ii<all_conts.size(); ++ii)
        {
            /* Copy contour coordinates */
            all_conts[ii] = MatrixType::Zero(polycont[ii].size(),2);
            for (std::size_t jj=0; jj<polycont[ii].size(); ++jj)
            {
                all_conts[ii](jj,0) = polycont[ii][jj].X;
                all_conts[ii](jj,1) = polycont[ii][jj].Y;
            }
        }

        /* Sort them by inclusion, and store which is hole and which not */
        polycont.is_hole = in_out_check(all_conts);

        /*-------------------------------------------*/
        /*       Transform coordinates from          */
        /*       middle contour to junctions         */
        /*-------------------------------------------*/
        for(std::size_t ii=0; ii<polycont.size(); ++ii)
            move2junctions(polycont[ii], junction_set[polycont[ii].orig_junc-1]);

        /*-------------------------------------------*/
        /*            Simplify points                */
        /*-------------------------------------------*/
        simplify(polycont, simplify_tol);

        /*-------------------------------------------*/
        /* Sort out clockwise, hole counterclockwise */
        /*-------------------------------------------*/
        clockwise_sort(polycont);
        
        
        /*-------------------------------------------*/
        /* Divide by 2, they should all be even      */
        /* numbers, because we are at junction points*/
        /*-------------------------------------------*/
        for(auto& path:polycont)
        {
            for(auto& pt:path)
            {
                assert(pt.X%2 == 0);
                assert(pt.Y%2 == 0);
                pt.X = pt.X/2;
                pt.Y = pt.Y/2;
            }
        }
        
        /* Erase the 'single pixels', they cause problems */
        /* TODO - Recode them better */
        std::list<std::size_t> to_erase;
        for(std::size_t ii=0; ii<polycont.size(); ++ii)
            if (polycont[ii].size()<=3)
                to_erase.push_front(ii);
        for(std::size_t ii:to_erase)
        {
            polycont.erase(polycont.begin()+ii);
            polycont.is_hole.erase(polycont.is_hole.begin()+ii);
        }
    }
        
    
    void polycont2mask(const ContContainer& polycont, MaskType& mask)
    {
        /* Allocate */
        mask = MaskType::Constant(polycont.im_sx, polycont.im_sy, false);
        
        /* Scan all points in the image and check if they are in or out */
        Point pt;
        for(pt.X = 1; pt.X<2*polycont.im_sx; pt.X+=2)
        {
            for(pt.Y = 1; pt.Y<2*polycont.im_sy; pt.Y+=2)
            {
                /* We count the parity of polygons that contain the point */
                int sign = -1;
                for(const auto& path: polycont)
                {
                    /* Multiply coordinates by 2, to avoid problems  */
                    /* exactly 'on' the border of the polygon        */
                    Path path2(path);
                    for(Point& pt2: path2)
                    {
                        pt2.X *=2;
                        pt2.Y *=2;
                    }
                    
                    /* Now check if it's 'completely in' */
                    if (ClipperLib::PointInPolygon(pt, path2)>0)
                        sign *= -1;
                }
                
                /* If odd, it's 'active' */
                if (sign>0)
                    mask((pt.X-1)/2,(pt.Y-1)/2) = true;
            }
        }
    }
    
}
#endif   
    
    