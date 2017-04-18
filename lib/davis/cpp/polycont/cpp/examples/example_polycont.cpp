#include <iostream>
#include "cpp/io.hpp"
#include "cpp/mask2polycont.hpp"
#include "cpp/poly_compare.hpp"

using namespace PolyCont;

int main()
{
    /*--------------------------------------------------*/
    /*  Create a simple mask with two connected         */
    /*  components and display its contour coordinates  */
    /*--------------------------------------------------*/
    MaskType mask(500,500);
    mask.block( 10,100,100,200) = MaskType::Constant(100, 200, true); // 10-110, 100-300
    mask.block(200,200,100,200) = MaskType::Constant(100, 200, true); // 200-300, 200-400

    /* Transform it to PolyCont */
    ContContainer conts;
    mask2polycont(mask,conts,3);
    
    /* Display */
    std::cout << "Mask with two connected components:" << std::endl;
    std::cout << conts;
    
    
    
    
    /*--------------------------------------------------*/
    /*  Create two simple masks and do some operations  */
    /*--------------------------------------------------*/
    
    /* Create mask1 */
    MaskType mask1 = MaskType::Constant(600, 600, false);
    mask1.block(100,100,300,300) = MaskType::Constant(300, 300, true); 
        
    /* Create mask2 */
    MaskType mask2 = MaskType::Constant(600, 600, false);
    mask2.block(200,200,300,300) = MaskType::Constant(300, 300, true); 
    
    /* Transform them to PolyCont */
    ContContainer conts1; mask2polycont(mask1,conts1,3);
    ContContainer conts2; mask2polycont(mask2,conts2,3);

    /* Display */   
    std::cout << "The two masks:" << std::endl;
    std::cout << conts1;
    std::cout << conts2;

    /* Get the intersection between polygons and area */
    double  int_polys = poly_intersection(conts1, conts2);
    double area_polys = poly_area(conts1);

    /* Display */
    std::cout << "Intersection and area:" << std::endl;

    std::cout << " - int_polys  = " << int_polys << std::endl;
    std::cout << " - area_polys = " << area_polys << std::endl;
    
    return 0;
}