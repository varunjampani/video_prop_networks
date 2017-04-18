// ------------------------------------------------------------------------
//  Copyright (C)
//  Federico Perazzi <perazzif@inf.ethz.ch>
//  April 2016
// ------------------------------------------------------------------------
#include <polycont/cpp/mask2poly.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python.hpp>
#include <boost/numpy.hpp>

// djikstra
#include <Eigen/Dense>

#include <boost/config.hpp>
#include <iostream>

#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/property_map/property_map.hpp>

namespace py = boost::python;
namespace np = boost::numpy;

#include <Eigen/Dense>
#include "cpp/mask2polycont.hpp"

using namespace std;
using namespace PolyCont;

//void mexFunction( int nlhs, mxArray *plhs[],
              //int nrhs, const mxArray*prhs[] )
//{

struct _ContContainer: public ContContainer
{
	py::list contour_coords;

};
_ContContainer _mask2poly(const np::ndarray& pymask, float tolerance) {

		np::dtype dtype = np::dtype::get_builtin<bool>();

		// Required transposition to match matlab impl
		np::ndarray _pymask = np::zeros(py::make_tuple(pymask.shape(1), pymask.shape(0)),
				np::dtype::get_builtin<bool>());

		// Transpose
		for(int i=0; i < _pymask.shape(0); ++i) {
			for(int j=0; j < _pymask.shape(1); ++j) {
				_pymask[i][j] = pymask[j][i];
			}
		}

		Eigen::Map<Eigen::Array<bool,Eigen::Dynamic,Eigen::Dynamic> >
						mask((bool*)_pymask.get_data(),_pymask.shape(1),_pymask.shape(0));

    /* Output contours */
    _ContContainer all_conts;

    /*-------------------------------------------*/
    /*         Call the actual function          */
    /*-------------------------------------------*/
    mask2polycont(mask, all_conts, tolerance);

		for (std::size_t ii=0; ii<all_conts.size(); ++ii) {

		dtype = np::dtype::get_builtin<int>();
			np::ndarray curr_poly = np::empty(
					py::make_tuple(all_conts[ii].size(),2),dtype);
			for (std::size_t jj=0; jj<all_conts[ii].size(); ++jj) {
				curr_poly[jj][0] = all_conts[ii][jj].Y;
				curr_poly[jj][1] = all_conts[ii][jj].X;
			}
			all_conts.contour_coords.append(curr_poly);
		}

		return all_conts;
}

np::ndarray _get_longest_cont(const py::list& contour_coords) {

	int max_id     = -1;
	int max_length = 0;

	for(int jj = 0; jj < py::len(contour_coords); ++jj) {
		np::ndarray arr = boost::python::extract<np::ndarray>(contour_coords[jj]);
		if(max_length <= arr.shape(0)) {
			max_id     = jj;
			max_length = arr.shape(0);
		}
	}
	return boost::python::extract<np::ndarray>(contour_coords[max_id]);;
}

np::ndarray _contour_upsample(const np::ndarray cont, float cont_th) {

	np::ndarray _cont = cont.astype(np::dtype::get_builtin<float>());

	struct Point {float x,y;};
	std::vector<Point> diff(_cont.shape(0)-1);

	const float* ptr_cont = (float*) _cont.get_data();

	for(int i=1; i < cont.shape(0); ++i) {
		diff[i-1].x = ptr_cont[i*2+0] - ptr_cont[(i-1)*2+0];
		diff[i-1].y = ptr_cont[i*2+1] - ptr_cont[(i-1)*2+1];
	}

	std::vector<float> nv(diff.size());

	// Compute length of each segment
	for(int i=0; i<diff.size(); ++i) {
		nv[i] = sqrt(diff[i].x*diff[i].x + diff[i].y*diff[i].y);
	}

	std::vector<Point> up_cont;
	up_cont.push_back(Point{ptr_cont[0],ptr_cont[1]});

	// Now upsample contour
	for(int i=0; i<nv.size(); ++i) {
		if(nv[i] > cont_th) {
			int n_segm = ceil((float)nv[i]/(float)cont_th);
			Point curr_point = up_cont.back();
			Point vec{diff[i].x/(float)n_segm,diff[i].y/(float)n_segm};

			for(int j=0; j < n_segm-1; ++j) {
				curr_point.x += vec.x;
				curr_point.y += vec.y;
				up_cont.push_back(curr_point);
			}
		}
		up_cont.push_back(Point{ptr_cont[(i+1)*2+0],ptr_cont[(i+1)*2+1]});
	}
	// Copy results
	np::ndarray res = np::zeros(py::make_tuple(up_cont.size(), 2),
				np::dtype::get_builtin<float>());

	for(int i = 0; i < up_cont.size(); ++i) {
		res[i][0] = up_cont[i].x;
		res[i][1] = up_cont[i].y;
	}
	return res;
}
// ------------------------------------------------------------------------
// Jordi Pont-Tuset - http://jponttuset.github.io/
// April 2016
// ------------------------------------------------------------------------
// This file is part of the DAVIS package presented in:
//   Federico Perazzi, Jordi Pont-Tuset, Brian McWilliams,
//   Luc Van Gool, Markus Gross, Alexander Sorkine-Hornung
//   A Benchmark Dataset and Evaluation Methodology for Video Object Segmentation
//   CVPR 2016
// Please consider citing the paper if you use this code.
// ------------------------------------------------------------------------

using namespace boost;
using namespace std;

/* Typedefs from Boost */
typedef adjacency_list < listS, vecS, directedS,
        no_property, property < edge_weight_t, double > > graph_t;
typedef graph_traits < graph_t >::vertex_descriptor vertex_descriptor;
typedef std::pair<int, int> Edge;

std::vector<vertex_descriptor> run_one_dijkstra(std::list<Edge>& edge_list, std::list<double>& edge_costs, const int num_nodes, int orig)
{
    /* Define the graph and the containers for the distances and parent nodes */
    graph_t g(edge_list.begin(), edge_list.end(), edge_costs.begin(), num_nodes);
    property_map<graph_t, edge_weight_t>::type weightmap = get(edge_weight, g);
    std::vector<vertex_descriptor> parents(num_vertices(g));
    std::vector<double>              dists(num_vertices(g));

    /* Run Dijkstra to all nodes */
    vertex_descriptor orig_vertex = vertex(orig, g);
    dijkstra_shortest_paths(g, orig_vertex,
            predecessor_map(boost::make_iterator_property_map(parents.begin(), get(vertex_index, g))).
            distance_map(boost::make_iterator_property_map(dists.begin(), get(vertex_index, g))));

    return parents;
}


py::tuple _match_dijkstra(const np::ndarray& prhs) {

		np::ndarray _prhs = np::zeros(py::make_tuple(prhs.shape(1),prhs.shape(0)),
				np::dtype::get_builtin<double>());

		// Transpose
		for(int i=0; i < _prhs.shape(0); ++i) {
			for(int j=0; j < _prhs.shape(1); ++j) {
				_prhs[i][j] = prhs[j][i];
			}
		}
    /* Cost matrix (doubles) */
    Eigen::Map<Eigen::Array<double,Eigen::Dynamic,Eigen::Dynamic> >
            costs_matrix((double *)_prhs.get_data(),_prhs.shape(1),_prhs.shape(0));
		Eigen::Array<double,Eigen::Dynamic,Eigen::Dynamic> costs(costs_matrix);

		//[> Sizes of the graph <]
		const int n1 = costs.rows();
		const int n2 = costs.cols();
		const int num_nodes = n1*n2;

		//[> Create a look-up table for node indices <]
		Eigen::Array<std::size_t,Eigen::Dynamic,Eigen::Dynamic> sub2ind(n1,n2);
		size_t curr_id = 0;
		for (size_t xx=0; xx<n1; ++xx)
				for (size_t yy=0; yy<n2; ++yy)
						sub2ind(xx,yy) = curr_id++;

		//[> Inverted LUT <]
		vector<pair<size_t,size_t>> ind2sub(num_nodes);
		for (size_t xx=0; xx<n1; ++xx)
				for (size_t yy=0; yy<n2; ++yy)
						ind2sub[sub2ind(xx,yy)] = make_pair(xx,yy);

		//[> Define the edges of the graph <]
		std::list<Edge> edge_list;
		std::list<double> edge_costs;
		for (std::size_t xx=0; xx<n1; ++xx)
		{
				for (std::size_t yy=0; yy<n2; ++yy)
				{
						//[> Down <]
						if (yy>0)
						{
								edge_list.emplace_back(sub2ind(xx,yy-1),sub2ind(xx,yy));
								edge_costs.emplace_back(costs(xx,yy));
						}
						//[> Left <]
						if (xx>0)
						{
								edge_list.emplace_back(sub2ind(xx-1,yy),sub2ind(xx,yy));
								edge_costs.emplace_back(costs(xx,yy));
						}

						//[> Down-left <]
						if ((yy>0) && (xx>0))
						{
								edge_list.emplace_back(sub2ind(xx-1,yy-1),sub2ind(xx,yy));
								edge_costs.emplace_back(costs(xx,yy));
						}
				}
		}


		//[> Origin and destination nodes - First we assume the 0-0 matching <]
		int orig = 0;
		int dest = num_nodes-1;
		std::vector<vertex_descriptor> parents = run_one_dijkstra(edge_list,edge_costs,num_nodes,orig);

	 //Get path to destination by scanning predecessors
	 //We also get the minimum-cost node
		std::vector<size_t> opt_path;
		size_t curr_predecessor = dest;
		double min_cost = numeric_limits<double>::max();
		size_t min_id = curr_predecessor;
		while(curr_predecessor!=orig)
		{
				double curr_cost = costs(ind2sub[curr_predecessor].first,ind2sub[curr_predecessor].second);
				if (curr_cost<min_cost)
				{
						min_cost = curr_cost;
						min_id = curr_predecessor;
				}
				opt_path.emplace_back(curr_predecessor);
				curr_predecessor = parents[curr_predecessor];
		}
		opt_path.emplace_back(curr_predecessor);
		int min_xx = ind2sub[min_id].first;
		int min_yy = ind2sub[min_id].second;

		np::ndarray outputMatrix = np::zeros(py::make_tuple(opt_path.size(),2),
				np::dtype::get_builtin<double>());

		//[> Output pairs - +1 for Matlab <]
		for (size_t i=0; i<opt_path.size(); i++) {
				outputMatrix[i][0] = ind2sub[opt_path[i]].first ;
				outputMatrix[i][1] = ind2sub[opt_path[i]].second;
		}

		return py::make_tuple(outputMatrix,min_xx,min_yy);
}

BOOST_PYTHON_MODULE(tstab) {
	// Initialize numpy
	np::initialize();

	py::class_<_ContContainer,std::shared_ptr<_ContContainer>>(
			"ContContainer")
		.def_readwrite("contour_coords",&_ContContainer::contour_coords)
		.def_readwrite("im_sx",&_ContContainer::im_sx)
		.def_readwrite("im_sy",&_ContContainer::im_sy)
		.def_readonly("is_hole",&_ContContainer::is_hole);

	py::def("mask2poly",_mask2poly);
	py::def("get_longest_cont",_get_longest_cont);
	py::def("contour_upsample",_contour_upsample);
	py::def("match_dijkstra",_match_dijkstra);

	boost::python::class_<std::vector<bool>>("PyVecBool")
			.def(boost::python::vector_indexing_suite<std::vector<bool>,true >());
}

