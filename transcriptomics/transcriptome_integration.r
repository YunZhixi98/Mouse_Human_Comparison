if(!require(Seurat))
  remotes::install_github("satijalab/seurat", "seurat5", quiet = TRUE)
if (!require(dplyr))
  install.packages("dplyr")
if (!require(SeuratObject))
  install.packages("SeuratObject")
if (!require(BPCells))
  remotes::install_github("bnprks/BPCells")
if (!require(lisi))
  devtools::install_github("immunogenomics/lisi")
if (!require("BiocManager", quietly = TRUE))
  install.packages("BiocManager")
if(!require(glmGamPoi))
  BiocManager::install("glmGamPoi")

library(Seurat)
library(dplyr)
library(SeuratObject)
library(BPCells)
library(ggplot2)
library(lisi)
library(glmGamPoi)

options(future.globals.maxSize = 3e9)

set.seed(821)

options(Seurat.object.assay.version = "v5")

h_path <- "G:/data/transcriptome/adata_h_to_bink.h5ad"
h_mat <- open_matrix_dir(dir = paste0(gsub(".h5ad", "", h_path), "_BP"))
h_object <- CreateSeuratObject(counts = h_mat, project = 'human')
h_object.obs <- read.csv(paste0(gsub(".h5ad", "", h_path), "_obs.csv"), row.names = 1)
h_object$batch <- h_object.obs$donor_id
h_object$species <- "human"
#h_object.var <- read.csv(paste0(gsub(".h5ad", "", h_path), "_var.csv"), row.names = 1)
h_object

m_path <- "G:/data/transcriptome/MERFISH/MH_m_merfish_isocortex_neurons.h5ad"
m_data <- open_matrix_anndata_hdf5(m_path)
# write_matrix_dir(
#     mat = m_data,
#     dir = paste0(gsub(".h5ad", "", m_path), "_BP"),
#     overwrite = T
# )
m_mat <- open_matrix_dir(dir = paste0(gsub(".h5ad", "", m_path), "_BP"))
m_object <- CreateSeuratObject(counts = m_mat, project = 'mouse')
m_object.obs <- read.csv(paste0(gsub(".h5ad", "", m_path), "_obs.csv"), row.names = 1)
#m_object.var <- read.csv(paste0(gsub(".h5ad", "", m_path), "_var.csv"), row.names = 1)
m_object$batch <- m_object.obs$donor_id
m_object$species <- "mouse"
m_object

excluded_gene <- read.csv("E:/ZhixiYun/Projects/fMOST_atlas/Data/gene/gene_symbol_group_id_homologous_excluded_genes.csv")$acronym

# Get the intersection of gene names between human and mouse datasets
common_genes <- intersect(rownames(h_object[["RNA"]]$counts), rownames(m_object[["RNA"]]$counts))
common_genes <- setdiff(common_genes, excluded_gene)

# Subset the human and mouse matrices to include only the common genes
h_object <- subset(h_object, features = common_genes)
m_object <- subset(m_object, features = common_genes)


# h_lower_nFeature <- quantile(h_object$nFeature_RNA, 0.001)
# h_lower_nCount <- quantile(h_object$nCount_RNA, 0.001)
# m_lower_nFeature <- quantile(m_object$nFeature_RNA, 0.001)
# m_lower_nCount <- quantile(m_object$nCount_RNA, 0.001)
# 
# # Filter cells
# # h_object <- subset(h_object, 
# #                    subset = nFeature_RNA > h_lower_nFeature & 
# #                      nCount_RNA > h_lower_nCount)
m_object <- subset(m_object, 
                  subset = nFeature_RNA >= 10 & 
                    nCount_RNA >= 10)



h_object[["RNA"]] <- split(h_object[["RNA"]], f = h_object$batch)
m_object[["RNA"]] <- split(m_object[["RNA"]], f = m_object$batch)

# m_object <- SCTransform(m_object, method = "glmGamPoi")
# h_object <- SCTransform(h_object)

# Merge two Seurat objects
merged_obj <- merge(x = m_object, y = h_object)
merged_obj


merged_obj <- NormalizeData(merged_obj, scale.factor = 1000)
merged_obj <- FindVariableFeatures(merged_obj, nfeatures = length(common_genes))



filtered_features_list <- list()
for (layer_name in Layers(merged_obj[["RNA"]], search = 'data')) {
  hvf_info <- HVFInfo(merged_obj[["RNA"]], layer = layer_name)
  non_zero_var_genes <- rownames(hvf_info)[hvf_info$variance != 0]
  print(length(non_zero_var_genes))
  filtered_features_list[[layer_name]] <- non_zero_var_genes
}
common_filtered_features <- Reduce(intersect, filtered_features_list)
print(paste("number of std!=0 genes:", length(common_filtered_features)))
merged_obj <- subset(merged_obj, features = common_filtered_features)



sketch_rate <- 0.10
ncells <- list()
for (layer_name in Layers(object = merged_obj[["RNA"]], search = 'data')) {
  layer_matrix <- GetAssayData(merged_obj, layer = layer_name)
  current_cell_count <- round(ncol(layer_matrix) * sketch_rate)
  ncells[[layer_name]] <- current_cell_count
}
ncells


# merged_obj <- SCTransform(merged_obj, method = "glmGamPoi", conserve.memory = TRUE)


sketch_object <- SketchData(
  object = merged_obj,
  ncells = ncells,
  method = "LeverageScore",
  sketched.assay = "sketch"
)
sketch_object

DefaultAssay(sketch_object) <- "sketch"
sketch_object <- FindVariableFeatures(sketch_object, verbose = F, nfeatures = length(common_genes))

sketch_object <- ScaleData(sketch_object, verbose = F)
sketch_object <- RunPCA(sketch_object, verbose = F)
sketch_object <- FindNeighbors(sketch_object, dims = 1:20)
sketch_object <- RunUMAP(sketch_object, dims = 1:20)


sketch_object@meta.data$species <- sketch_object@meta.data$orig.ident
sketch_object@meta.data$species[sketch_object@meta.data$species != 'mouse'] <- 'human'

sketch_object <- IntegrateLayers(
  object = sketch_object,
  method = CCAIntegration,
  orig.reduction = "pca",
  new.reduction = "integrated.cca",
#  normalization.method = 'SCT',
  dims = 1:20,
  k.anchor = 30,
  verbose = FALSE
)

sketch_object <- FindNeighbors(sketch_object, reduction = "integrated.cca", dims = 1:20)
sketch_object <- FindClusters(sketch_object,
                              resolution = 1,
                              cluster.name = "cca_clusters")

sketch_object <- RunUMAP(
  sketch_object,
  reduction = "integrated.cca",
  dims = 1:20,
  reduction.name = "umap.cca"
)

lisi_score <- compute_lisi(sketch_object@reductions$umap.cca@cell.embeddings, 
                           sketch_object@meta.data[rownames(sketch_object@reductions$umap.cca@cell.embeddings), ], c("species", "batch"))
ggplot(data=lisi_score, aes(x = batch, y = species)) +
  geom_hex(bins = 50) +
  geom_abline(slope = 1, linetype = "dashed") +
  labs(title = "LISI Score Distribution")



p1 <- DimPlot(
  sketch_object,
  reduction = "umap.cca",
  alpha = 0.5,
  group.by = c("batch", "species", "cca_clusters"),
  combine = FALSE,
  label.size = 2
)
ggsave("./p1_1_dimplot.png", plot = p1[[1]], width = 10, height = 8)
ggsave("./p1_2_dimplot.png", plot = p1[[2]], width = 10, height = 8)
ggsave("./p1_3_dimplot.png", plot = p1[[3]], width = 10, height = 8)


# -------------------
# project Integration
# -------------------


sketch_object <- ProjectIntegration(
  object = sketch_object,
  sketched.assay = "sketch",
  assay = "RNA",
  reduction = "integrated.cca"
)
sketch_object <- ProjectData(
  object = sketch_object,
  sketched.assay = "sketch",
  assay = "RNA",
  sketched.reduction = "integrated.cca.full",
  full.reduction = "integrated.cca.full",
  dims = 1:20,
  refdata = list(celltype.full = "cca_clusters")
)


sketch_object <- RunUMAP(
  sketch_object,
  reduction = "integrated.cca.full",
  dims = 1:20,
  reduction.name = "umap.full",
  reduction.key = "UMAP_full_"
)



write.csv(sketch_object@meta.data, 
          "./sketch_object_meta_data.csv")
write.csv(Embeddings(sketch_object$'pca'),
          './sketch_object_pca.csv')
write.csv(Embeddings(sketch_object$'umap'),
          './sketch_object_umap.csv')
write.csv(Embeddings(sketch_object$'umap.cca'),
          './sketch_object_umap_cca.csv')
write.csv(Embeddings(sketch_object$'integrated.cca.full'),
          './sketch_object_cca_full.csv')
write.csv(Embeddings(sketch_object$'umap.full'),
          './sketch_object_umap_full.csv')


DimPlot(
  sketch_object,
  reduction = "umap.full",
  group.by = c("species", "celltype.full"),
  combine = F,
  alpha = 0.1
)
