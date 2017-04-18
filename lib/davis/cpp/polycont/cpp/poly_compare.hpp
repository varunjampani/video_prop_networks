// ------------------------------------------------------------------------
//  Copyright (C)
//  Jordi Pont-Tuset <jponttuset@vision.ee.ethz.ch>
//  July 2015
// ------------------------------------------------------------------------
#ifndef POLYCOMPARE_HPP
#define POLYCOMPARE_HPP

#include <iostream>
#include <cpp/containers.hpp>

namespace PolyCont
{
    double poly_intersection(const ContContainer& poly1, const ContContainer& poly2)
    {
        /* ************* TEMPORAL ************
         * Conversion, we should remove junctions from container
         * or define it for our containers */
        ClipperLib::Paths paths1(poly1.begin(),poly1.end());
        ClipperLib::Paths paths2(poly2.begin(),poly2.end());
        
        /* Get the intersection polygon */
        ClipperLib::Clipper clpr;
        clpr.AddPaths(paths1, ClipperLib::ptSubject, true);
        clpr.AddPaths(paths2, ClipperLib::ptClip   , true);
        ClipperLib::Paths solution;
        clpr.Execute(ClipperLib::ctIntersection, solution, ClipperLib::pftEvenOdd, ClipperLib::pftEvenOdd);
        
        /* Get its area */
        double int_area = 0;
        for(std::size_t ii=0; ii<solution.size(); ++ii)
            int_area += std::abs(ClipperLib::Area(solution[ii]));
        
        return int_area;
    }

    double poly_area(const ContContainer& poly)
    {
        /* Get its area */
        double area = 0;
        for(std::size_t ii=0; ii<poly.size(); ++ii)
            area += ClipperLib::Area(poly[ii]);
        
        return std::abs(area);
    }
}          
#endif