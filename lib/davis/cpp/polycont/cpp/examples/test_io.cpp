#include <iostream>
#include "cpp/io.hpp"

using namespace PolyCont;

int test_one_mask(MaskType mask)
{
    /************ Write it to file in binary  ***************/
    std::string filename("cpp/examples/test_io.bin");
    write_mask(filename, mask, IOBinary);

    /* Now read it */
    MaskType mask2;
    read_mask(filename, mask2, IOBinary);
        
    /******* Write it to file as polygonal contours  ********/
    filename = "cpp/examples/test_io.pcont";
    write_mask(filename, mask, IOPolyCont);

    /* Now read it */
    MaskType mask3;
    read_mask(filename, mask3, IOPolyCont);
        
    return (mask==mask2) + 2*(mask==mask3);
}


int main()
{
    /* Create a set of masks */
    std::vector<MaskType> masks;
    
    /* Simple two blobs */
    MaskType mask1(600,500);
    mask1.block( 50, 50,100,200) = MaskType::Constant(100, 200, true); //  50-150, 150-250
    mask1.block(200,200,100,200) = MaskType::Constant(100, 200, true); // 200-300, 200-400
    masks.push_back(mask1);
    
    
    /* Simple with a hole */
    MaskType mask2(500,600);
    mask2.block( 80,100,300,300) = MaskType::Constant(300, 300, true); 
    mask2.block(150,150,200,200) = MaskType::Constant(200, 200, true); 
    masks.push_back(mask2);
    
    /* Touching borders */
    masks.push_back(MaskType::Constant(5, 10, true));

    /* Test all masks and show results */
    std::size_t ii = 1;
    for(const MaskType& mask: masks)
    {
        int result = test_one_mask(mask);
        if (result%2)
            std::cout << "Binary   I/O on mask " << ii << " successful!" << std::endl;
        else
            std::cout << "Binary   I/O on mask " << ii << " FAILED!" << std::endl;

        if (result>=2)
            std::cout << "PolyCont I/O on mask " << ii << " successful!" << std::endl;
        else
            std::cout << "PolyCont I/O on mask " << ii << " FAILED!" << std::endl;
        ++ii;
    }
    
    return 0;
}