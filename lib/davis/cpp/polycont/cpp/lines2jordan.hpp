// ------------------------------------------------------------------------
//  Copyright (C)
//  Jordi Pont-Tuset <jponttuset@vision.ee.ethz.ch>
//  July 2015
// ------------------------------------------------------------------------

#ifndef LINES2JORDAN_HPP
#define LINES2JORDAN_HPP

#include <vector>
#include <set>
#include <unordered_map>
#include <cpp/containers.hpp>

namespace PolyCont
{

    void append_coords(Path& piece, const Path& cont_to_add, int last_junction)
    {
        if (cont_to_add.orig_junc==last_junction)
        {        
            piece.end_junc = cont_to_add.end_junc;

            /* Add the coordinates in the order they are */
            for(std::size_t ii=0; ii<cont_to_add.size(); ++ii)
                piece.push_back(cont_to_add[ii]);
        }
        else if (cont_to_add.end_junc==last_junction)
        {
            piece.end_junc = cont_to_add.orig_junc;

            /* Invert the coordinates */
            for(std::size_t ii=0; ii<cont_to_add.size(); ++ii)
                piece.push_back(cont_to_add[cont_to_add.size()-ii-1]);
        }
        else
        {
            assert(0);
            //mexErrMsgTxt("append_coords: Something went wrong");
        }
    // mexPrintf(" (junc %d to %d)\n", last_junction, piece.end_junc);                   

    }

    int get_other_extreme(const Path& contour, int junction_id)
    {
        if (junction_id==contour.orig_junc)
            return contour.end_junc;
        else           
            return contour.orig_junc;
    }



    Point get_first_next_coord(const Path& contour, int junction_id)
    {
        if (contour.orig_junc==junction_id)
            return contour[0];
        else
            return contour[contour.size()-1];
    }

    int add_piece(int cycle_junction, const ContContainer& lines, std::list<std::pair<int,int>>& swept_junctions_and_conts, std::set<int>& swept_junctions, ContContainer& jordan)
    {
    // mexPrintf("--- Found a cycle around junction %d\n",cycle_junction);                   

        /* Find it in the list */
        auto list_it = swept_junctions_and_conts.begin();
        for( ; list_it!=swept_junctions_and_conts.end(); ++list_it)
            if (list_it->first == cycle_junction)
                break;

        assert(list_it!=swept_junctions_and_conts.end());
        //    mexErrMsgTxt("Something went wrong");

        /* Container for the Jordan curve */
        Path curr_piece;
        curr_piece.orig_junc = cycle_junction;

        /* Sweep until we cycle */
        for( ; list_it!=swept_junctions_and_conts.end(); )
        {
    // mexPrintf("Adding contour %d", list_it->second);                   

            append_coords(curr_piece, lines[list_it->second], list_it->first);
            auto to_delete = list_it;
            ++list_it;

            /* Erase them from the list and set */
            swept_junctions.erase(to_delete->first);
            swept_junctions_and_conts.erase(to_delete);
        }

        /* Sanity check */
        assert(curr_piece.end_junc==cycle_junction);
        //    mexErrMsgTxt("Something went wrong");

        /* Add the cycle to the results container */
        jordan.push_back(curr_piece);

        /* Return the new cont_to_explore as the last one that was explored */
        int new_nest_cont_to_explore = -1;
        auto list_it2 = swept_junctions_and_conts.rbegin();
        if (list_it2!=swept_junctions_and_conts.rend())
            new_nest_cont_to_explore = list_it2->second;
        return new_nest_cont_to_explore;
    }


    void lines2jordan(const ContContainer& lines,
                      const std::vector<Junction>& junctions,
                      ContContainer& jordan)
    {
    // /* Display junctions to debug */
    // mexPrintf("Junction set:\n");
    // for(const auto& junc: junctions)
    // {
    //     mexPrintf("%d (%d,%d): ", (int)junc.id, (int)junc.pos.X, (int)junc.pos.Y);
    //     for(auto id: junc.cont_pieces)
    //         mexPrintf("%d, ", (int)id);
    //     mexPrintf("\n");
    // }
    //     
    // /* Display contours to debug */
    // mexPrintf("Contour set:\n");
    // std::size_t ii=0;
    // for(const auto& line: lines)
    // {
    //     mexPrintf("%d (%d,%d)\n", ii, (int)line.orig_junc, (int)line.end_junc);
    //     ++ii;
    // }

        /* Create the set of contours to be explored */
        std::set<int> conts_to_sweep;
        for(int ii=0; ii<lines.size(); ++ii)
            conts_to_sweep.insert(ii);

        /* Lines yet to be explored on each junction */
        std::unordered_map<int,std::set<int>> conts_on_junct(junctions.size());
        for(const auto& junc: junctions)
        {
            auto& curr_list = conts_on_junct[junc.id];
            for(auto id: junc.cont_pieces)
                curr_list.insert(id);
        }

        /* Sweep all contours */
        while(not conts_on_junct.empty())
        {

    // mexPrintf("Contours to explore on each junction:\n");
    // for(const auto& it2: conts_on_junct)
    // {
    //     mexPrintf("%d: ", it2.first);
    //     for(auto id: it2.second)
    //         mexPrintf("%d, ", (int)id);
    //     mexPrintf("\n");
    // }
            /* Containers for currently swept junctions and contours */
            std::list<std::pair<int,int>> swept_junctions_and_conts;
            std::set<int> swept_junctions;

            /* Pick one contour yet to explore */
            int junct_to_explore = conts_on_junct.begin()->first;
            int  cont_to_explore = *(conts_on_junct[junct_to_explore].begin());

    // mexPrintf("A: Exploring junc %d cont %d\n", junct_to_explore, cont_to_explore);

            /* Erase the contour from the list and the junction if  *
             * all the incident contours have been explored         */
            /* From origin junction */
            conts_on_junct[junct_to_explore].erase(cont_to_explore); 
            if (conts_on_junct[junct_to_explore].empty())
                conts_on_junct.erase(junct_to_explore);
            /* From end junction */
            int end_junc = get_other_extreme(lines[cont_to_explore], junct_to_explore);
            conts_on_junct[end_junc].erase(cont_to_explore); 
            if (conts_on_junct[end_junc].empty())
                conts_on_junct.erase(end_junc);

    // mexPrintf("Contours to explore on each junction:\n");
    // for(const auto& it2: conts_on_junct)
    // {
    //     mexPrintf("%d: ", it2.first);
    //     for(auto id: it2.second)
    //         mexPrintf("%d, ", (int)id);
    //     mexPrintf("\n");
    // }

            /* Add it to the set */
            swept_junctions.insert(junct_to_explore);
            swept_junctions_and_conts.push_back(std::make_pair(junct_to_explore,cont_to_explore));

            /* Pick the next junction to explore */
            int next_junct_to_explore = get_other_extreme(lines[cont_to_explore], junct_to_explore);

            /* Do we have a cycle? Add it! */
            if (swept_junctions.count(next_junct_to_explore)) /* We found a cycle */ 
                add_piece(next_junct_to_explore, lines, swept_junctions_and_conts, swept_junctions, jordan);

            /* Sweep until we explore all connections from that piece of contour */
            while(not swept_junctions.empty())
            {
                /* Next junction */
                junct_to_explore = next_junct_to_explore;

    // mexPrintf("B: Exploring junc %d cont %d\n", junct_to_explore, cont_to_explore);

                /* Pick one contour yet to explore from next_junct_to_explore,
                 * avoiding going in 'straight' line */
                auto it_junct = conts_on_junct[junct_to_explore].begin();

                Point next_first = get_first_next_coord(lines[*it_junct]      , junct_to_explore);
                Point curr_last  = get_first_next_coord(lines[cont_to_explore], junct_to_explore);
    // mexPrintf(" - Going from (%d,%d) to (%d,%d)\n", curr_last.X, curr_last.Y, next_first.X, next_first.Y);

                if( (next_first.X==curr_last.X) or (next_first.Y==curr_last.Y) )
                    ++it_junct;
                assert(it_junct!=conts_on_junct[junct_to_explore].end());
                    //mexErrMsgTxt("Something went wrong: it_junct==conts_on_junct[junct_to_explore].end()");
                cont_to_explore = *it_junct;

                /* Erase the contour from the list and the junction if  *
                 * all the incident contours have been explored         */
                /* From origin junction */
                conts_on_junct[junct_to_explore].erase(cont_to_explore); 
                if (conts_on_junct[junct_to_explore].empty())
                    conts_on_junct.erase(junct_to_explore);
                /* From end junction */
                end_junc = get_other_extreme(lines[cont_to_explore], junct_to_explore);
                if (conts_on_junct.count(end_junc))
                {
                    conts_on_junct[end_junc].erase(cont_to_explore); 
                    if (conts_on_junct[end_junc].empty())
                        conts_on_junct.erase(end_junc);
                }

                /* Add it to the explored list */
                swept_junctions.insert(junct_to_explore);
                swept_junctions_and_conts.push_back(std::make_pair(junct_to_explore,cont_to_explore));

                /* Get the next to explore */
                next_junct_to_explore = get_other_extreme(lines[cont_to_explore], junct_to_explore);

                /* Do we have a cycle? Add it! */
                if (swept_junctions.count(next_junct_to_explore))
                    cont_to_explore = add_piece(next_junct_to_explore, lines, swept_junctions_and_conts, swept_junctions, jordan);
            }
        }
    }
}

#endif


