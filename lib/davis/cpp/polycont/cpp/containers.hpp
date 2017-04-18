// ------------------------------------------------------------------------
//  Copyright (C)
//  Jordi Pont-Tuset <jponttuset@vision.ee.ethz.ch>
//  July 2015
// ------------------------------------------------------------------------
#ifndef CONTAINERS_HPP
#define CONTAINERS_HPP

#include <Eigen/Dense>
#include <vector>
#include <set>
#include <map>
#include <list>
#include <fstream>

#include <clipper/clipper.hpp>


namespace PolyCont
{
    /* Our main container for double matrices */
    typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> MatrixType;

    /* Container for boolean masks */
    typedef Eigen::Matrix<bool,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> MaskType;
        
    /* Type to avoid copies when reading from Mat */
    typedef Eigen::Map<Eigen::MatrixXd> eigen_map;

    /* Container of a coordinate */
    typedef ClipperLib::IntPoint Point;

    /* Container of a junction */
    struct Junction
    {
        Junction(std::size_t new_id, std::size_t new_x,
                 std::size_t new_y)
        : id(new_id), pos(new_x,new_y), cont_pos()
        {};

        /* Label */
        std::size_t id;

        /* Position */
        Point pos;

        /* Set of coordinate of the contours that form the junction */
        std::vector<Point> cont_pos;

        /* Set of ids of the incident contour pieces */
        std::set<double> cont_pieces;
    };

    /* Container of a piece of contour (from junction to junction) */
    struct Path : public ClipperLib::Path
    {
        Path()
        : orig_junc(), end_junc(), ClipperLib::Path()
        {};

        Path(std::size_t new_orig_junc, Point first_point)
        : orig_junc(new_orig_junc), end_junc(), ClipperLib::Path(1,first_point)
        {};

        /* Labels of the junctions */
        std::size_t orig_junc;
        std::size_t end_junc;
    };
    
    /* Container of a full contour (set of contour pieces) */
    struct ContContainer : public std::vector<Path>
    {
        ContContainer(std::size_t size=0) : std::vector<Path>(size)
        {}
        
        std::size_t im_sx;
        std::size_t im_sy;
        std::vector<bool> is_hole;
        
        friend std::ostream& operator<<(std::ostream& stream, const ContContainer& conts)
        {
            for(const Path& cont: conts)
            {
                for(const Point& pt: cont)
                    stream << pt;
                stream << std::endl;
            }   

            return stream;
        }
    };
}
#endif
