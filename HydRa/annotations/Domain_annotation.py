def plot_annotations_v2(ax1, ax3, ax4, RBP_id, IDR_coords, low_complexity_coords, seqdir='/home/wjin/projects/CLIP_seq/HYDRA_eCLIP/Data2/candidate_ORFs/protein_sequences/', Domain_dir='/home/wjin/projects/CLIP_seq/HYDRA_eCLIP/Data2/candidate_ORFs/Domain_search_output_collapsed/'):
    """
    ax1: ax for domains
    ax2: ax for RBDpeps
    IDR_coords: list of integers, presenting the coordinates of IDRs.
    low_complexity_coords: list of integers, presenting the coordinates of amino acids within low complexity regions.
    """
    cm = plt.cm.get_cmap('Pastel1')
    seqfile=os.path.join(seqdir, RBP_id+'.fasta')
    Domain_file=os.path.join(Domain_dir, RBP_id+'.domain.out.colapsed.Evalue0.1.0.01.out')
    
    if os.path.exists(seqfile):
        f=open(seqfile)
        seq=f.read().split('\n')[1]
        f.close()
        bar1=np.zeros(len(seq)) #bar for domains
        #bar2=np.zeros(len(seq)) #bar for RBDpeps
        bar3=np.zeros(len(seq)) #bar for IDR
        bar4=np.zeros(len(seq)) #bar for low complexity
        ## add domain annotation from Pfam searches
        if os.path.exists(Domain_file):
            try:
                df=pd.read_table(Domain_file, header=None)
                domain_list=list(df[0])
                domain_coords=np.array(df[[7,8]])
                domains=zip(domain_list, domain_coords.tolist())
                for dom, (start, end) in domains:
                    start=start-1  #Convert to 0-based coordinates
                    end=end-1  #Convert to 0-based coordinates
                    if dom in RBDs:
                        bar1[start:end+1]=-1 #pink
                    else:
                        bar1[start:end+1]=-6 #light green
            except pd.errors.EmptyDataError:
                 domain_list=[]
                    
#         ## add RBDpeps annotations
#         if RBP_id in set(RBDpep_grouped.groups.keys()):
#             RBDpep_coords=RBDpep_table.loc[RBDpep_grouped.groups[RBP_id]][['Start','End']].values
#             for RBDpep_coord in RBDpep_coords: ## PS: RBDpep_coords are in 1-based coordinates system, which is different from python's which is 0-based. Also, End is included in the range.
#                 Start=RBDpep_coord[0]-1 #Convert to 0-based coordinates
#                 End=RBDpep_coord[1]-1 #Convert to 0-based coordinates
#                 bar2[Start:End+1]=-8 #red
            
        for IDR_coord in IDR_coords:
            bar3[IDR_coord]=-2
            
        for low_complexity_coord in low_complexity_coords:
            bar4[low_complexity_coord]=-5
            
        bar1=np.array([bar1]*max(int(len(seq)/80),1))
        #bar2=np.array([bar2]*max(int(len(seq)/80),1))
        bar3=np.array([bar3]*max(int(len(seq)/80),1))
        bar4=np.array([bar4]*max(int(len(seq)/80),1))
        im1=ax1.imshow(bar1, cmap=cm, vmin=-8, vmax=0)
        #im2=ax2.imshow(bar2, cmap=cm, vmin=-8, vmax=0)
        im3=ax3.imshow(bar3, cmap=cm, vmin=-8, vmax=0)
        im4=ax4.imshow(bar4, cmap=cm, vmin=-8, vmax=0)
        ax1.set_title(RBP_id+'_domains',fontsize='x-large',verticalalignment='bottom')
        #ax2.set_title(RBP_id+'_RBDpeps',fontsize='x-large',verticalalignment='bottom')
        ax3.set_title(RBP_id+'_IDR',fontsize='x-large',verticalalignment='bottom')
        ax4.set_title(RBP_id+'_LowComplexity',fontsize='x-large',verticalalignment='bottom')
        
        if len(domain_list)>0:
            for dom, (start, end) in zip(domain_list,domain_coords.tolist()):
                ax1.text((start+end)/2, 0.5*int(len(seq)/100), dom, fontsize='large', horizontalalignment='center',verticalalignment='center')

        ax1.axes.get_yaxis().set_visible(False)
        #ax2.axes.get_yaxis().set_visible(False)
        ax3.axes.get_yaxis().set_visible(False)
        ax4.axes.get_yaxis().set_visible(False)