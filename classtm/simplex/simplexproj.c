/*
 #  File            : simplexproj.c
 #
 #  Version History : 0.1, May 31, 2017
 #
 #  Author          : Nozomu Okuda
 #
 #  Description     : This file contains an implementation in the C language
 #                    of Condat's algorithm, described in the research paper:
 #
 #                    L. Condat, "Fast Projection onto the Simplex and the
 #                    l1 Ball", preprint Hal-01056171, 2014.
 #
 #                    This implementation comes with no warranty: due to the
 #                    limited number of tests performed, there may remain
 #                    bugs. In case the functions would not do what they are
 #                    supposed to do, please email the author (contact info
 #                    to be found on the web).
 #
 #                    If you use this code or parts of it for any purpose,
 #                    the author asks you to cite the paper above or, in
 #                    that event, its published version. Please email Condat if
 #                    the proposed algorithms were useful for one of your
 #                    projects, or for any comment or suggestion.
 #
 #                    The simplexproj function was originally implemented by
 #                    Condat; Okuda modified Condat's original code to suit his
 #                    purposes.
 #
 #  Usage rights    : Copyright Laurent Condat.
 #                    This file is distributed under the terms of the CeCILL
 #                    licence (compatible with the GNU GPL), which can be
 #                    found at the URL "http://www.cecill.info".
 #
 #  This software is governed by the CeCILL license under French law and
 #  abiding by the rules of distribution of free software. You can  use,
 #  modify and or redistribute the software under the terms of the CeCILL
 #  license as circulated by CEA, CNRS and INRIA at the following URL :
 #  "http://www.cecill.info".
 #
 #  As a counterpart to the access to the source code and rights to copy,
 #  modify and redistribute granted by the license, users are provided only
 #  with a limited warranty  and the software's author,  the holder of the
 #  economic rights,  and the successive licensors  have only  limited
 #  liability.
 #
 #  In this respect, the user's attention is drawn to the risks associated
 #  with loading,  using,  modifying and/or developing or reproducing the
 #  software by the user in light of its specific status of free software,
 #  that may mean  that it is complicated to manipulate,  and  that  also
 #  therefore means  that it is reserved for developers  and  experienced
 #  professionals having in-depth computer knowledge. Users are therefore
 #  encouraged to load and test the software's suitability as regards their
 #  requirements in conditions enabling the security of their systems and/or
 #  data to be ensured and,  more generally, to use and operate it in the
 #  same conditions as regards security.
 #
 #  The fact that you are presently reading this means that you have had
 #  knowledge of the CeCILL license and that you accept its terms.
*/

/* The following functions are implemented:
simplexproj (proposed algorithm)
All these functions take the same parameters. They project the vector y onto
the closest vector x of same length (parameter N in the paper) with x[n]>=0,
n=0..N-1, and sum_{n=0}^{N-1}x[n]=a.
We can have x==y (projection done in place). If x!=y, the arrays x and y must
not overlap, as x is used for temporary calculations before y is accessed.
We must have length>=1 and a>0.
*/

#include <stdlib.h>


/* Proposed algorithm */
/* y is the input vector that is to be projected
 * x is the output vector where the projected y will be written
 * length is the length of y (and of x)
 * a is the maximum value for the dimension
 */
void simplexproj(double* y, double* x,
const unsigned int length, const double a) {
    // if the input vector is the same as the output vector, we need to make a
    // place to keep temporary information
    double*    aux = (x==y ? (double*)malloc(length*sizeof(double)) : x);
    double*  aux0=aux;
    int        auxlength=1;
    int        auxlengthold=-1;
    // not only set initial tau but also put the first value into the list of
    // values likely to be greater than the final tau (called v in the paper)
    double    tau=(*aux=*y)-a;
    int     i=1;
    for (; i<length; i++)
        if (y[i]>tau) {
            // make the updates to the data structures while also making the
            // check to see if values need to belong to the other temporary
            // list, \tilde{v}; note that aux is being used to hold both v and
            // \tilde{v}, their separation being kept straight by auxlength and
            // auxlengthold
            if ((tau+=((aux[auxlength]=y[i])-tau)/(auxlength-auxlengthold))
            <=y[i]-a) {
                tau=y[i]-a;
                auxlengthold=auxlength-1;
            }
            // note that auxlength gets incremented only when the current value
            // is greater than the current tau
            auxlength++;
        }
    if (auxlengthold>=0) {
        // the following two lines delineates where \tilde{v} ends and v starts
        auxlength-=++auxlengthold;
        aux+=auxlengthold;
        // the statement following the if statement is not only updating tau,
        // it is also adding the elements in \tilde{v} that belong into v
        while (--auxlengthold>=0)
            if (aux0[auxlengthold]>tau)
                tau+=((*(--aux)=aux0[auxlengthold])-tau)/(++auxlength);
    }
    do {
        auxlengthold=auxlength-1;
        // again, tricky manipulation of auxlength to keep track of what values
        // in aux belong to v; the statement following the if statement copies
        // values from the old v into the next iteration's v
        for (i=auxlength=0; i<=auxlengthold; i++)
            if (aux[i]>tau)
                aux[auxlength++]=aux[i];
            else
                tau+=(tau-aux[i])/(auxlengthold-i+auxlength);
    } while (auxlength<=auxlengthold);
    for (i=0; i<length; i++)
        x[i]=(y[i]>tau ? y[i]-tau : 0.0);
    if (x==y) free(aux0);
}
