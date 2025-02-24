import molsysmt as msm
from enmt import pyunitwizard as puw
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm
from copy import deepcopy
from sklearn.linear_model import LinearRegression
import nglview as nv
import lindelint as ldl


class AnisotropicNetworkModel():

    def __init__(self, molecular_system, selection='atom_name=="CA"', structure_index=0, cutoff='9 angstroms',
                 stiffness=None, syntax='MolSysMT'):

        self.molecular_system = msm.convert(molecular_system, to_form="molsysmt.MolSys",
                                            structure_indices=structure_index)
 
        self.atom_indices = None
        self.n_nodes = 0
        self.cutoff = None
        self.stiffness = None
        self.contacts = None
        self.hessian_matrix = None
        self.eigenvalues = None
        self.eigenvectors = None
        self.frequencies = None
        self.modes = None 
        self.b_factors = None
        self.b_factors_exp = None
        self.scaling_factor = None
        self.sqrt_deviation = None
        self.correlation_matrix = None    
        self.inverse = None

        self.make_model(selection=selection, cutoff=cutoff, syntax=syntax)

    def make_model(self, selection='atom_name=="CA"', cutoff='9 angstroms', syntax='MolSysMT'):

        if selection is not None:
            self.atom_indices = msm.select(self.molecular_system, selection=selection, syntax=syntax)

        if cutoff is not None:
            self.cutoff = puw.standardize(cutoff)

        # contacts

        contacts = msm.structure.get_contacts(self.molecular_system,
                                              selection=self.atom_indices,
                                              structure_indices=0,
                                              threshold=self.cutoff)

        self.contacts = contacts[0]

        self.n_nodes = self.contacts.shape[0]

        np.fill_diagonal(self.contacts, False)

        # Hessian matrix: eigenvals, eigenvects

        self.hessian_matrix = np.zeros((3*self.n_nodes, 3*self.n_nodes), dtype=float)

        coordinates = msm.get(self.molecular_system, element='atom', selection=self.atom_indices, structure_indices=0, coordinates=True)
        coordinates = puw.get_value(coordinates[0])

        for ii in range(self.n_nodes):
            iii = ii*3
            for jj in range(ii):
                if self.contacts[ii,jj]:
                    jjj = jj*3

                    rij = coordinates[ii]-coordinates[jj]
                    distance2 = np.dot(rij, rij)

                    for kk in range(3):
                        for gg in range(3):

                            val_aux = -(coordinates[jj][kk]-coordinates[ii][kk])*(coordinates[jj][gg]-coordinates[ii][gg])/distance2

                            ## Sub matrix Hij
                            self.hessian_matrix[iii+kk, jjj+gg]=val_aux
                            self.hessian_matrix[jjj+gg, iii+kk]=val_aux
                        
                            #Sub matrix Hii:
                            self.hessian_matrix[iii+kk, iii+gg]-=val_aux
                            self.hessian_matrix[jjj+gg, jjj+kk]-=val_aux

        self.eigenvalues, self.eigenvectors = np.linalg.eigh(self.hessian_matrix)

        # Frequencies and modes

        self.modes = np.zeros(shape=(self.n_nodes*3, self.n_nodes, 3), dtype=float)

        for aa in range(self.n_nodes*3):
            for ii in range(self.n_nodes):
                iii=ii*3
                for jj in range(3):
                    self.modes[aa,ii,jj]=self.eigenvectors[iii+jj,aa]

        self.modes = self.modes[6:]
        self.frequencies = self.eigenvalues[6:]

    def show_contact_map(self):

        plt.matshow(self.contacts, cmap='binary')
        return plt.show()

    def show_mode(self, mode=0, signed_amplitude=True, show_zero_line=True):

        aux_mode = self.modes[mode]
        amplitude_residue = np.sqrt(aux_mode[:,0]**2 + aux_mode[:,1]**2 + aux_mode[:,2]**2)

        if signed_amplitude:

            ref_residue = np.argmax(amplitude_residue)
            ref_vector = aux_mode[ref_residue]
            dot_product = np.einsum('ij,j->i', aux_mode, ref_vector)
            signs = np.sign(dot_product)
            amplitude_residue = amplitude_residue*signs

        plt.plot(amplitude_residue)

        if show_zero_line:
            plt.axhline(0, ls='--', c='gray')

        plt.show()

    def view(self, protein=True, network=False, color_by_value=None, representation='cartoon', cmap='bwr'):

        if protein:
            output = msm.view(self.molecular_system)


            if color_by_value is not None:

                output.clear()

                msm.thirds.nglview.set_color_by_value(output, color_by_value, element='group', representation=representation, cmap=cmap)

        else:
            output = nv.NGLWidget()

        if network:

            coordinates = msm.get(self.molecular_system, element='atom', selection=self.atom_indices, coordinates=True)
            coordinates = puw.get_value(coordinates[0], to_unit='angstroms')

            for ii in range(self.n_nodes):
                for jj in range(ii+1, self.n_nodes):
                    if self.contacts[ii,jj]:

                        kwargs = {'position1':coordinates[ii].tolist(),
                                  'position2':coordinates[jj].tolist(),
                                  'color': [0.6, 0.6, 0.6],
                                  'radius': [0.2]}
                        
                        msg = output._get_remote_call_msg("addBuffer",
                                            target="Widget",
                                            args=["cylinder"],
                                            kwargs=kwargs,
                                            fire_embed=True)

                        def callback(widget, msg=msg):
                            widget.send(msg)

                        callback._method_name = 'addBuffer'
                        callback._ngl_msg = msg

                        output._ngl_displayed_callbacks_before_loaded.append(callback)

                        output._ngl_msg_archive.append(msg)
        

        return output


    def view_mode(self, mode=0, amplitude='6.0 angstroms', oscillation_steps=60, method='LinDelInt',
                  representation='cartoon', color_by_amplitude=False, color_by_signed_amplitude=False, cmap='bwr',
                  arrows=False, color_arrows='#808080', radius_arrows='0.2 angstroms'):

        if oscillation_steps>0:

            molecular_system = self.trajectory_along_mode(mode=mode, amplitude=amplitude,
                                                          oscillation_steps=oscillation_steps, method=method)
            view = msm.view(molecular_system)

        else:

            view = msm.view(self.molecular_system)

        if color_by_amplitude or color_by_signed_amplitude:

            aux_mode = self.modes[mode]
            amplitude_residue = np.sqrt(aux_mode[:,0]**2 + aux_mode[:,1]**2 + aux_mode[:,2]**2)

            if color_by_signed_amplitude:

                ref_residue = np.argmax(amplitude_residue)
                ref_vector = aux_mode[ref_residue]
                dot_product = np.einsum('ij,j->i', aux_mode, ref_vector)
                signs = np.sign(dot_product)
                amplitude_residue = amplitude_residue*signs

            msm.thirds.nglview.set_color_by_value(view, amplitude_residue, element='group',
                                                  representation=representation, mid_value=0.0,
                                                  cmap=cmap)

        if arrows:

            origin_arrows = msm.get(self.molecular_system, element='atom', selection=self.atom_indices, coordinates=True)
            arrows = puw.quantity(100.0*self.modes[mode], 'angstroms')
            end_arrows = origin_arrows + arrows

            msm.thirds.nglview.add_arrows(view, origin_arrows, end_arrows,
                                          color=color_arrows, radius=radius_arrows)

        return view


    def trajectory_along_mode(self, selection='all', mode=0, amplitude='6.0 angstroms', oscillation_steps=60,
                              keep_box=False, method='LinDelInt', syntax='MolSysMT'):

        # method in ['LinDelInt', 'physical']

        if method=='LinDelInt':

            coordinates_nodes = msm.get(self.molecular_system, element='atom', selection=self.atom_indices,
                                        coordinates=True)
            mode_array = self.modes[mode]

            interpolator = ldl.Interpolator(puw.get_value(coordinates_nodes[0]), mode_array)

            components_involved = msm.get(self.molecular_system, element='component', selection='atom_index in @self.atom_indices',
                                          component_index=True)
            atoms_involved = msm.get(self.molecular_system, element='atom', selection='component_index in @components_involved',
                                     atom_index=True)
            atom_indices = msm.select(self.molecular_system, element='atom', selection=selection, mask=atoms_involved, syntax=syntax)

            molecular_system = msm.extract(self.molecular_system, selection=atom_indices)
            coordinates = msm.get(molecular_system, element='atom', selection='all', coordinates=True)
            direction_mode = interpolator.do_your_thing(puw.get_value(coordinates[0]))
            direction_mode = direction_mode/np.linalg.norm(direction_mode)
            max_norm_atom = -1.0
            for ii in range(direction_mode.shape[0]):
                norm_atom = np.linalg.norm(direction_mode[ii,:])
                if max_norm_atom<norm_atom:
                    max_norm_atom=norm_atom
            factor = puw.quantity(amplitude)/max_norm_atom
            delta_f=2.0*np.pi/(oscillation_steps*1.0)
            new_coordinates = puw.quantity(np.zeros([oscillation_steps, coordinates.shape[1], coordinates.shape[2]], dtype=float), 'nm')
            for frame in range(oscillation_steps):
                new_coordinates[frame,:,:]=coordinates+factor*np.sin(delta_f*frame)*direction_mode

            molecular_system = msm.remove(molecular_system, structure_indices=0)
            msm.append_structures(molecular_system, new_coordinates)

            if keep_box==False:
                msm.set(molecular_system, element='system', box=None)

            return molecular_system

