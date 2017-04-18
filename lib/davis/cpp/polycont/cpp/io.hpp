// ------------------------------------------------------------------------
//  Copyright (C)
//  Jordi Pont-Tuset <jponttuset@vision.ee.ethz.ch>
//  July 2015
// ------------------------------------------------------------------------

#ifndef IO_HPP
#define IO_HPP

#include <cpp/containers.hpp>
#include <cpp/mask2polycont.hpp>

#include <fstream>
#include <string> 
#include <sstream>

/* Code to read/write a matrix to binary without any compression */
namespace PolyCont
{
    
    
    void write_polycont(const std::string& filename, const ContContainer& polycont)
    {
        /* Open filestream */
        std::fstream fs(filename, std::fstream::out);
        
        /* Write image size */
        fs << polycont.im_sx << "," << polycont.im_sy;
                        
        /* Write blobs separated with '/' */
        for(const auto& path:polycont)
        {
            fs << "/";

            /* All but last to file, to avoid an extra ',' */
            for(std::size_t ii=0; ii<path.size()-1; ++ii)
                fs << path[ii].X << "," << path[ii].Y << ",";
            
            /* Write last */
            fs << path.back().X << "," << path.back().Y;
        }

        fs.close();
    }
    
    
    void read_polycont(const std::string& filename, ContContainer& polycont)
    {
        /* Open filestream */   
        std::fstream fs(filename, std::fstream::in);

        /* Get size */
        std::string sx, sy;
        std::getline(fs, sx, ',');
        std::getline(fs, sy, '/');
        
        /* 'Allocate' */
        polycont.im_sx = std::stoi(sx);
        polycont.im_sy = std::stoi(sy);
        
        /* Separate the object into blobs */
        std::string path_str;
        while(std::getline(fs, path_str, '/'))
        {
            std::istringstream ss(path_str);
            std::string coord_str;
            bool is_x = true;
            Path  path;
            Point pt;
            while(std::getline(ss, coord_str, ','))
            {
                if (is_x)
                    pt.X = std::stoi(coord_str);
                else
                {
                    pt.Y = std::stoi(coord_str);
                    path.push_back(pt);
                }

                is_x = !is_x;
            }
            
            polycont.push_back(path);
        }
        
        /* Fill in the hole indicator */
        polycont.is_hole.resize(polycont.size());
        for(std::size_t ii=0; ii<polycont.size(); ++ii)
            polycont.is_hole[ii] = !is_clockwise(polycont[ii]);
    }
    
    
    
    
    
    
    template<class Matrix>
    void write_mask_polycont(const std::string& filename, const Matrix& matrix, double tolerance=3)
    {
        /* Transform it to PolyCont */
        ContContainer polycont;
        mask2polycont(matrix,polycont,tolerance);
        
        /* Write it as polycont */
        write_polycont(filename, polycont);
    }
    
    template<class Matrix>
    void read_mask_polycont(const std::string& filename, Matrix& matrix)
    {
        /* Read it as polycont */
        ContContainer polycont;
        read_polycont(filename, polycont);
        
        /* Transform it to Mask */
        polycont2mask(polycont,matrix);
    }
    
    
    
    
    template<class Matrix>
    void write_mask_binary(const std::string& filename, const Matrix& matrix)
    {
        std::ofstream out(filename, std::ios::out | std::ios::binary | std::ios::trunc);
        typename Matrix::Index rows=matrix.rows(), cols=matrix.cols();
        out.write((char*) (&rows), sizeof(typename Matrix::Index));
        out.write((char*) (&cols), sizeof(typename Matrix::Index));
        out.write((char*) matrix.data(), rows*cols*sizeof(typename Matrix::Scalar) );
        out.close();
    }
    
    template<class Matrix>
    void read_mask_binary(const std::string& filename, Matrix& matrix)
    {
        std::ifstream in(filename, std::ios::in | std::ios::binary);
        typename Matrix::Index rows=0, cols=0;
        in.read((char*) (&rows),sizeof(typename Matrix::Index));
        in.read((char*) (&cols),sizeof(typename Matrix::Index));
        matrix.resize(rows, cols);
        in.read( (char *) matrix.data() , rows*cols*sizeof(typename Matrix::Scalar) );
        in.close();
    }

    
    
    
    
    enum IOFormat { IOPolyCont, IOBinary };
    
    
    
    void write_mask(const std::string& filename, const MaskType& mask, IOFormat format=IOPolyCont, double tolerance=3)
    {
        switch(format)
        {
            case IOPolyCont: write_mask_polycont(filename, mask, tolerance);  break;
            case IOBinary  : write_mask_binary  (filename, mask);  break;
        }
    }
    
    
    void read_mask(const std::string& filename, MaskType& mask, IOFormat format=IOPolyCont)
    {
        switch(format)
        {
            case IOPolyCont: read_mask_polycont(filename, mask);  break;
            case IOBinary  : read_mask_binary  (filename, mask);  break;
        }
    }
}

#endif


