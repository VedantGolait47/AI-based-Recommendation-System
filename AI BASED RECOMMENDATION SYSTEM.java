import org.apache.mahout.cf.taste.eval.RecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.eval.AverageAbsoluteDifferenceRecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.impl.neighborhood.NearestNUserNeighborhood;
import org.apache.mahout.cf.taste.impl.recommender.GenericItemBasedRecommender;
import org.apache.mahout.cf.taste.impl.recommender.GenericUserBasedRecommender;
import org.apache.mahout.cf.taste.impl.similarity.PearsonCorrelationSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.neighborhood.UserNeighborhood;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.similarity.ItemSimilarity;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;

import java.io.File;
import java.util.List;

public class RecommendationSystem {
    public static void main(String[] args) {
        try {
            // Load data from CSV file
            DataModel model = new FileDataModel(new File("data.csv")); // Replace with your file path
            
            // User-based recommender
            UserSimilarity userSimilarity = new PearsonCorrelationSimilarity(model);
            UserNeighborhood neighborhood = new NearestNUserNeighborhood(2, userSimilarity, model);
            GenericUserBasedRecommender userBasedRecommender = new GenericUserBasedRecommender(model, neighborhood, userSimilarity);

            System.out.println("User-based recommendations:");
            List<RecommendedItem> userRecommendations = userBasedRecommender.recommend(1, 3); // Recommend for user 1, 3 products
            for (RecommendedItem recommendation : userRecommendations) {
                System.out.println(recommendation);
            }

            // Item-based recommender
            ItemSimilarity itemSimilarity = new PearsonCorrelationSimilarity(model);
            GenericItemBasedRecommender itemBasedRecommender = new GenericItemBasedRecommender(model, itemSimilarity);

            System.out.println("\nItem-based recommendations:");
            List<RecommendedItem> itemRecommendations = itemBasedRecommender.recommend(1, 3); // Recommend for user 1, 3 products
            for (RecommendedItem recommendation : itemRecommendations) {
                System.out.println(recommendation);
            }

            // Evaluate the recommender system
            RecommenderEvaluator evaluator = new AverageAbsoluteDifferenceRecommenderEvaluator();
            double score = evaluator.evaluate(
                    userBasedRecommender,
                    null,
                    model,
                    0.7, // Training set ratio
                    1.0  // Evaluation set ratio
            );
            System.out.println("\nEvaluation score: " + score);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
